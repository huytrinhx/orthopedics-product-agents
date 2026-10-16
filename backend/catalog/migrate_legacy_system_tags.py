"""One-off script carrying every document's legacy System tag over to a
Product Family and Trays (issue #12). Delete it, with legacy_systems.py, in
the legacy-removal ticket, once it has run in production.

Run it after the master catalog seed (backend/ingestion/
seed_master_catalog.py) has mirrored the families/trays into Postgres --
that's why this is a script rather than an Alembic data migration:

    python -m catalog.migrate_legacy_system_tags          # dry run
    python -m catalog.migrate_legacy_system_tags --apply  # write it

Each tagged document's system name goes through the explicit legacy map
(legacy_systems.resolve_family_name) to a family, and the document gets
that family plus *all* of the family's trays; a document whose family has
more than one tray is flagged in the report so an admin can trim the set
later in the tag modal. Unmapped and untagged documents are left without a
family (the indexing gate reports them as incomplete). Documents that
already have a family are skipped, so a re-run is a no-op and never undoes
a tray set someone has since trimmed.

The dry run (the default) prints the report and writes nothing. `--apply`
writes, per document: the Neo4j Document node's family/trays properties
(if the document has been indexed and so has a node), then the Postgres
family, tray set, and chunk family in one transaction. Graph first, so a
failure partway leaves that document still family-less in Postgres and the
next run picks it up again. Chunks are re-pointed in place, not
re-embedded.
"""
import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass, field

from catalog.legacy_systems import resolve_family_name
from catalog.repository import ProductFamilyRecord, TrayRecord, list_families_with_trays
from documents.repository import DocumentRecord, list_documents, set_family_and_trays
from retrieval.graph_client import GraphClient, get_graph_client


@dataclass
class DocumentMigration:
    document: DocumentRecord
    family: ProductFamilyRecord
    trays: list[TrayRecord]

    @property
    def needs_tray_review(self) -> bool:
        return len(self.trays) > 1


@dataclass
class MigrationPlan:
    documents_per_tag: Counter = field(default_factory=Counter)
    family_per_tag: dict[str, ProductFamilyRecord | None] = field(default_factory=dict)
    migrations: list[DocumentMigration] = field(default_factory=list)
    unmapped: list[DocumentRecord] = field(default_factory=list)
    untagged: list[DocumentRecord] = field(default_factory=list)
    already_migrated: list[DocumentRecord] = field(default_factory=list)


def plan_migration(
    documents: list[DocumentRecord], families: list[ProductFamilyRecord]
) -> MigrationPlan:
    """Pure: decides what each document gets, writes nothing. Family names
    match case-insensitively, the same way the legacy `systems` tags
    themselves are unique."""
    families_by_lowercase_name = {family.name.lower(): family for family in families}
    plan = MigrationPlan()
    for document in documents:
        if document.system_name:
            plan.documents_per_tag[document.system_name] += 1
            family = families_by_lowercase_name.get(
                resolve_family_name(document.system_name).lower()
            )
            plan.family_per_tag[document.system_name] = family
        else:
            family = None

        if document.product_family_id is not None:
            plan.already_migrated.append(document)
        elif not document.system_name:
            plan.untagged.append(document)
        elif family is None:
            plan.unmapped.append(document)
        else:
            plan.migrations.append(
                DocumentMigration(document=document, family=family, trays=list(family.trays))
            )
    return plan


def format_report(plan: MigrationPlan) -> list[str]:
    lines = ["Legacy system tags (documents per tag -> Product Family):"]
    for tag, count in sorted(plan.documents_per_tag.items()):
        family = plan.family_per_tag[tag]
        target = family.display_name if family else "UNMAPPED"
        lines.append(f"  {tag}: {count} -> {target}")

    lines.append(f"Documents to migrate: {len(plan.migrations)}")
    for migration in plan.migrations:
        tray_names = ", ".join(tray.display_name for tray in migration.trays) or "(no trays)"
        flag = f"  [REVIEW: {len(migration.trays)} trays]" if migration.needs_tray_review else ""
        lines.append(
            f"  {migration.document.filename} [{migration.document.system_name}] -> "
            f"{migration.family.display_name}; trays: {tray_names}{flag}"
        )

    needing_review = [migration for migration in plan.migrations if migration.needs_tray_review]
    lines.append(
        f"Documents whose family has more than one tray (trim later in the tag modal): "
        f"{len(needing_review)}"
    )
    for migration in needing_review:
        lines.append(
            f"  {migration.document.filename} -> {migration.family.display_name} "
            f"({len(migration.trays)} trays)"
        )

    unmapped_tags = sorted({document.system_name for document in plan.unmapped})
    lines.append(
        f"Unmapped tags (documents left without family/trays): {len(plan.unmapped)} documents"
        + (f" under {', '.join(unmapped_tags)}" if unmapped_tags else "")
    )
    for document in plan.unmapped:
        lines.append(f"  {document.filename} [{document.system_name}]")

    lines.append(f"Untagged documents (left without family/trays): {len(plan.untagged)}")
    for document in plan.untagged:
        lines.append(f"  {document.filename}")

    lines.append(f"Already have a family (skipped): {len(plan.already_migrated)}")
    for document in plan.already_migrated:
        lines.append(f"  {document.filename} -> {document.product_family_display_name}")
    return lines


async def apply_plan(plan: MigrationPlan, graph: GraphClient) -> None:
    for migration in plan.migrations:
        document_id = migration.document.id
        await graph.set_document_family_and_trays(
            str(document_id),
            migration.family.name,
            [tray.name for tray in migration.trays],
        )
        await set_family_and_trays(
            document_id, migration.family.id, [tray.id for tray in migration.trays]
        )


async def migrate(*, apply: bool, documents: list[DocumentRecord] | None = None) -> MigrationPlan:
    """`documents` defaults to every document; tests pass their own."""
    if documents is None:
        documents = await list_documents()
    plan = plan_migration(documents, await list_families_with_trays())
    for line in format_report(plan):
        print(line)
    if apply:
        await apply_plan(plan, get_graph_client())
        print(f"Applied: {len(plan.migrations)} documents tagged with a family and trays.")
    else:
        print("Dry run -- nothing written. Re-run with --apply to write it.")
    return plan


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true", help="write the changes (default: dry run)"
    )
    arguments = parser.parse_args()
    await migrate(apply=arguments.apply)


if __name__ == "__main__":
    asyncio.run(main())

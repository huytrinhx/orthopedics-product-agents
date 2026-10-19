"""Raw SQL against the `product_families` and `trays` mirror tables
(backend/migrations/versions/..._create_product_families_and_trays.py).

Neo4j stays the source of truth for the catalog hierarchy; these tables
mirror its Families and Trays into Postgres so documents can later be tagged
against them with real foreign keys. Both are written only by the master
catalog seed (backend/ingestion/seed_master_catalog.py), keyed on the stable
catalog name. `display_name` starts out as that catalog name and is never
touched by a re-seed after the row exists -- it's the admin-editable label
(the "Do Not Edit" master file has typos, e.g. `Ankle Fracture Plating
Sytem`, that get fixed here rather than in the file). A family's
`description_override` is the same kind of admin edit: the intent
classifiers' description of it, in place of the generated one.
"""
import uuid
from dataclasses import dataclass, field

from config.db import get_connection


@dataclass
class TrayRecord:
    id: uuid.UUID
    name: str
    display_name: str


@dataclass
class ProductFamilyRecord:
    id: uuid.UUID
    name: str
    display_name: str
    trays: list[TrayRecord] = field(default_factory=list)
    # The admin's description for the intent classifiers; None means the
    # generated one (catalog/family_descriptions.py).
    description_override: str | None = None


async def upsert_families_and_trays(trays_by_family: dict[str, list[str]]) -> None:
    """One transaction for the whole catalog, so a seed that fails partway
    never leaves a family mirrored without its trays. A tray already
    mirrored under another family is moved to the one given here -- the
    catalog decides which family a tray belongs to, not the first seed that
    happened to see it.
    """
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            for family_name, tray_names in trays_by_family.items():
                # The no-op `SET name = EXCLUDED.name` is there so RETURNING
                # still yields the existing row's id on a conflict --
                # `DO NOTHING` returns no row at all.
                await cursor.execute(
                    "INSERT INTO product_families (name, display_name) VALUES (%s, %s) "
                    "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
                    "RETURNING id",
                    (family_name, family_name),
                )
                (family_id,) = await cursor.fetchone()
                for tray_name in tray_names:
                    await cursor.execute(
                        "INSERT INTO trays (name, display_name, product_family_id) "
                        "VALUES (%s, %s, %s) "
                        "ON CONFLICT (name) DO UPDATE "
                        "SET product_family_id = EXCLUDED.product_family_id",
                        (tray_name, tray_name, family_id),
                    )
        await connection.commit()
    finally:
        await connection.close()


async def list_families_with_trays() -> list[ProductFamilyRecord]:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT id, name, display_name, description_override FROM product_families "
                "ORDER BY display_name"
            )
            families = [
                ProductFamilyRecord(family_id, name, display_name, description_override=override)
                for family_id, name, display_name, override in await cursor.fetchall()
            ]
            await cursor.execute(
                "SELECT id, name, display_name, product_family_id FROM trays ORDER BY display_name"
            )
            tray_rows = await cursor.fetchall()
    finally:
        await connection.close()

    families_by_id = {family.id: family for family in families}
    for tray_id, name, display_name, family_id in tray_rows:
        families_by_id[family_id].trays.append(TrayRecord(tray_id, name, display_name))
    return families


# ---- Admin display-name edits (issue #15) ----
#
# A blank or missing display name falls back to the catalog name rather than
# being stored empty, so `display_name` stays NOT NULL and every reader can
# use it as-is. `name` is never written here -- it's the seed's key.

_DISPLAY_NAME_OR_CATALOG_NAME = "COALESCE(NULLIF(btrim(%s), ''), name)"


async def set_family_display_name(
    family_id: uuid.UUID, display_name: str | None
) -> ProductFamilyRecord | None:
    """Returns the updated family (its `trays` left empty), or None if no
    family has that id."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"UPDATE product_families SET display_name = {_DISPLAY_NAME_OR_CATALOG_NAME} "
                "WHERE id = %s RETURNING id, name, display_name",
                (display_name, family_id),
            )
            row = await cursor.fetchone()
        await connection.commit()
    finally:
        await connection.close()
    return ProductFamilyRecord(*row) if row else None


async def set_tray_display_name(tray_id: uuid.UUID, display_name: str | None) -> TrayRecord | None:
    """Returns the updated tray, or None if no tray has that id."""
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"UPDATE trays SET display_name = {_DISPLAY_NAME_OR_CATALOG_NAME} "
                "WHERE id = %s RETURNING id, name, display_name",
                (display_name, tray_id),
            )
            row = await cursor.fetchone()
        await connection.commit()
    finally:
        await connection.close()
    return TrayRecord(*row) if row else None


async def set_family_description_override(
    family_id: uuid.UUID, description: str | None
) -> ProductFamilyRecord | None:
    """Sets the admin's description of a family for the intent classifiers.
    A blank or missing one is stored as NULL, which means the generated
    description again. Returns the updated family (its `trays` left empty),
    or None if no family has that id.
    """
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE product_families SET description_override = NULLIF(btrim(%s), '') "
                "WHERE id = %s RETURNING id, name, display_name, description_override",
                (description, family_id),
            )
            row = await cursor.fetchone()
        await connection.commit()
    finally:
        await connection.close()
    if row is None:
        return None
    family_id, name, display_name, description_override = row
    return ProductFamilyRecord(family_id, name, display_name, description_override=description_override)

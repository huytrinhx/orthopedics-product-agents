"""Thin wrapper around the Neo4j driver, pointed at AuraDB by default (see
ARCHITECTURE decision: AuraDB now, self-hosted Neo4j-on-AKS is a config
change later — swap the URI/credentials, not this client).

Schema (see CONTEXT.md's glossary for the domain story behind these):

    (Tray)-[:BELONGS_TO_FAMILY]->(ProductFamily)
    (Part)-[:BELONGS_TO_TRAY]->(Tray)
    (Part)-[:LOCATED_IN]->(TraySection)              -- shape only; unpopulated
                                                          until tray-overhead-
                                                          guide extraction exists
    (Part)-[:COMPATIBLE_WITH]->(Part)                -- e.g. plate <-> screw family
    (Part)-[:REQUIRES_TOOL]->(Part)                  -- e.g. screw <-> guidewire/driver
    (Part)-[:DIFFERENTIATES_FROM {explanation}]->(Part)
    (Procedure)-[:REQUIRES]->(Tray)
    (Part|Procedure)-[:SOURCED_FROM]->(Document)
    (Term)-[:ALIAS_OF|ABBREVIATION_OF]->(CanonicalTerm)

`Part` identity is its SKU and only its SKU (backend/evals/unite-master-csv.txt's
`Item No.`) — no name-based or fuzzy merging. Two ingestion paths write here:
backend/ingestion/seed_master_catalog.py (deterministic, the authoritative
Part/Tray/ProductFamily catalog) and backend/ingestion/entity_extraction.py
(LLM-based, prose facts attached to Parts the seed already created).
"""
import asyncio
import os

from neo4j import AsyncGraphDatabase

_RELATIONSHIP_TYPES = (
    "BELONGS_TO_FAMILY",
    "BELONGS_TO_TRAY",
    "LOCATED_IN",
    "COMPATIBLE_WITH",
    "REQUIRES_TOOL",
    "DIFFERENTIATES_FROM",
    "REQUIRES",
    "SOURCED_FROM",
    "ALIAS_OF",
    "ABBREVIATION_OF",
)


class GraphClient:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def ensure_constraints(self) -> None:
        """Uniqueness constraints doubling as the schema declaration — Neo4j
        has no separate DDL step, so this is what `alembic upgrade head` is
        for Postgres. Idempotent; safe to call on every seed-script run.
        """
        statements = [
            "CREATE CONSTRAINT part_sku IF NOT EXISTS FOR (p:Part) REQUIRE p.sku IS UNIQUE",
            "CREATE CONSTRAINT tray_name IF NOT EXISTS FOR (t:Tray) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT family_name IF NOT EXISTS FOR (f:ProductFamily) REQUIRE f.name IS UNIQUE",
            "CREATE CONSTRAINT procedure_name IF NOT EXISTS FOR (pr:Procedure) REQUIRE pr.name IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT canonical_term_name IF NOT EXISTS FOR (c:CanonicalTerm) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT term_name IF NOT EXISTS FOR (t:Term) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT tray_section_key IF NOT EXISTS FOR (s:TraySection) REQUIRE s.key IS UNIQUE",
        ]
        async with self._driver.session() as session:
            for statement in statements:
                await session.run(statement)

    # ---- Master catalog seed (backend/ingestion/seed_master_catalog.py) ----

    async def upsert_product_family(self, name: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                "MERGE (:ProductFamily {name: $name})", name=name
            )

    async def upsert_tray(self, name: str, product_family: str | None) -> None:
        async with self._driver.session() as session:
            if product_family:
                await session.run(
                    """
                    MERGE (t:Tray {name: $name})
                    MERGE (f:ProductFamily {name: $family})
                    MERGE (t)-[:BELONGS_TO_FAMILY]->(f)
                    """,
                    name=name,
                    family=product_family,
                )
            else:
                await session.run("MERGE (:Tray {name: $name})", name=name)

    async def upsert_part(self, sku: str, tray: str, **properties: str | int | None) -> None:
        clean_props = {k: v for k, v in properties.items() if v is not None}
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (p:Part {sku: $sku})
                SET p += $properties
                WITH p
                MERGE (t:Tray {name: $tray})
                MERGE (p)-[:BELONGS_TO_TRAY]->(t)
                """,
                sku=sku,
                tray=tray,
                properties=clean_props,
            )

    async def upsert_compatible_with(self, sku_a: str, sku_b: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH (a:Part {sku: $sku_a})
                MATCH (b:Part {sku: $sku_b})
                MERGE (a)-[:COMPATIBLE_WITH]->(b)
                """,
                sku_a=sku_a,
                sku_b=sku_b,
            )

    async def upsert_requires_tool(self, sku: str, tool_sku: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH (p:Part {sku: $sku})
                MATCH (tool:Part {sku: $tool_sku})
                MERGE (p)-[:REQUIRES_TOOL]->(tool)
                """,
                sku=sku,
                tool_sku=tool_sku,
            )

    # ---- Synonym seed (backend/ingestion/seed_synonyms.py) ----

    async def upsert_synonym_cluster(
        self, canonical: str, terms: list[str], notes: str | None = None
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (c:CanonicalTerm {name: $canonical})
                SET c.notes = coalesce($notes, c.notes)
                WITH c
                UNWIND $terms AS term_name
                MERGE (t:Term {name: term_name})
                MERGE (t)-[:ALIAS_OF]->(c)
                """,
                canonical=canonical,
                terms=terms,
                notes=notes,
            )

    async def upsert_abbreviation(self, canonical: str, abbreviation: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (c:CanonicalTerm {name: $canonical})
                MERGE (t:Term {name: $abbreviation})
                MERGE (t)-[:ABBREVIATION_OF]->(c)
                """,
                canonical=canonical,
                abbreviation=abbreviation,
            )

    # ---- Prose extraction writes (backend/ingestion/entity_extraction.py) ----

    async def upsert_document(
        self, document_id: str, filename: str, doc_type: str | None, system: str | None
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.filename = $filename, d.doc_type = $doc_type, d.system = $system
                """,
                id=document_id,
                filename=filename,
                doc_type=doc_type,
                system=system,
            )

    async def attach_differentiation(
        self, sku_a: str, sku_b: str, explanation: str, document_id: str
    ) -> bool:
        """Only writes the edge if both SKUs already exist as `Part` nodes
        (i.e. came from the master catalog seed) — prose extraction attaches
        to known parts, it never mints new ones. Returns whether it wrote
        anything, so callers can tell a hallucinated/unknown SKU from a
        successful attach.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (a:Part {sku: $sku_a})
                MATCH (b:Part {sku: $sku_b})
                MERGE (a)-[r:DIFFERENTIATES_FROM]->(b)
                SET r.explanation = $explanation
                WITH a, b
                MATCH (d:Document {id: $document_id})
                MERGE (a)-[:SOURCED_FROM]->(d)
                MERGE (b)-[:SOURCED_FROM]->(d)
                RETURN a.sku AS attached
                """,
                sku_a=sku_a,
                sku_b=sku_b,
                explanation=explanation,
                document_id=document_id,
            )
            return await result.single() is not None

    async def attach_procedure(self, procedure: str, tray: str, document_id: str) -> bool:
        """Only writes if `tray` already exists (from the master catalog
        seed) — a hallucinated tray name is dropped rather than minting a
        new, unanchored Tray node.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Tray {name: $tray})
                MERGE (p:Procedure {name: $procedure})
                MERGE (p)-[:REQUIRES]->(t)
                WITH p
                MATCH (d:Document {id: $document_id})
                MERGE (p)-[:SOURCED_FROM]->(d)
                RETURN p.name AS attached
                """,
                tray=tray,
                procedure=procedure,
                document_id=document_id,
            )
            return await result.single() is not None

    async def document_exists(self, document_id: str) -> bool:
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (d:Document {id: $id}) RETURN d.id AS id", id=document_id
            )
            return await result.single() is not None

    # ---- Lookups used to keep the LLM prompt scoped to real SKUs/trays ----

    async def list_parts_for_family(self, product_family: str) -> list[dict]:
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (p:Part)-[:BELONGS_TO_TRAY]->(:Tray)-[:BELONGS_TO_FAMILY]->(f:ProductFamily {name: $family})
                RETURN p.sku AS sku, p.description AS description
                """,
                family=product_family,
            )
            return [dict(record) async for record in result]

    async def list_trays_for_family(self, product_family: str) -> list[str]:
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Tray)-[:BELONGS_TO_FAMILY]->(:ProductFamily {name: $family})
                RETURN t.name AS name
                """,
                family=product_family,
            )
            return [record["name"] async for record in result]

    # ---- Query surface (backend/agents/tools/graph_query.py + synonym_resolve.py) ----

    async def query_related_entities(
        self, entity: str, relationship: str | None = None
    ) -> list[dict]:
        if relationship is not None and relationship not in _RELATIONSHIP_TYPES:
            return []
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e)
                WHERE toLower(coalesce(e.sku, '')) = toLower($entity)
                   OR toLower(coalesce(e.name, '')) = toLower($entity)
                   OR toLower(coalesce(e.description, '')) CONTAINS toLower($entity)
                WITH e
                ORDER BY CASE WHEN toLower(coalesce(e.sku, e.name, '')) = toLower($entity) THEN 0 ELSE 1 END
                LIMIT 1
                CALL (e) {
                    MATCH (e)-[r]->(related)
                    WHERE $relationship IS NULL OR type(r) = $relationship
                    RETURN type(r) AS relationship, 'outgoing' AS direction, r, related
                    UNION
                    MATCH (related)-[r]->(e)
                    WHERE $relationship IS NULL OR type(r) = $relationship
                    RETURN type(r) AS relationship, 'incoming' AS direction, r, related
                }
                RETURN relationship, direction, labels(related) AS related_labels,
                       coalesce(related.sku, related.name, related.id) AS related_entity,
                       properties(related) AS related_properties,
                       properties(r) AS relationship_properties
                """,
                entity=entity,
                relationship=relationship,
            )
            return [dict(record) async for record in result]

    async def get_synonym_groups(self) -> list[list[str]]:
        """Canonical synonym clusters, source of truth for query-time expansion
        via backend/agents/tools/synonym_resolve.py. Each group includes the
        canonical name itself plus every ALIAS_OF/ABBREVIATION_OF term —
        callers don't need to distinguish the two relationship types."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (c:CanonicalTerm)
                OPTIONAL MATCH (t:Term)-[:ALIAS_OF|ABBREVIATION_OF]->(c)
                RETURN c.name AS canonical, collect(t.name) AS variants
                """
            )
            groups = []
            async for record in result:
                group = [record["canonical"], *[v for v in record["variants"] if v]]
                groups.append(group)
            return groups

    async def close(self) -> None:
        await self._driver.close()


# Neo4j's async driver must be created and used on the same running event
# loop. In production there's exactly one (uvicorn's) for the app's whole
# lifetime, so a single cached instance would suffice -- but this process can
# also run multiple loops in the same interpreter (e.g. FastAPI's TestClient
# driving one loop while other async tests run on pytest-asyncio's session
# loop), which a plain @lru_cache singleton would silently bind to whichever
# loop called it first, breaking every other loop. Cache one client per loop
# instead.
_clients_by_loop: dict[asyncio.AbstractEventLoop, GraphClient] = {}


def get_graph_client() -> GraphClient:
    loop = asyncio.get_running_loop()
    client = _clients_by_loop.get(loop)
    if client is None:
        client = GraphClient(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"],
        )
        _clients_by_loop[loop] = client
    return client

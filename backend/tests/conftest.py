"""Points the test suite at the disposable postgres-test/neo4j-test containers
(docker-compose.yml's "test" profile) instead of the dev instances the rest of
this repo's tooling uses -- so running pytest locally never writes fixture
data into the database you're doing manual dev/testing against (see
build-log.md: an earlier run of this suite against the dev DB left ~1300
Neo4j nodes and ~1500 Postgres rows of test fixtures mixed into real data).

CI already gets this for free -- its postgres/neo4j services are spun up
fresh per run and torn down after -- so this override only applies locally;
CI sets `CI` and keeps using the DATABASE_URL/NEO4J_* the workflow already
provides. Bring the local test containers up once with
`docker compose --profile test up -d` before running pytest.

The containers themselves are NOT torn down between separate local `pytest`
invocations, though -- a test that fails before reaching its own cleanup
step (an assertion error before a `client.delete(...)`, say) leaves rows
behind, and those accumulate run over run. Seen for real, repeatedly: enough
leftover `systems` rows made detect_intent's system-disambiguation guardrail
fire on ordinary, unrelated queries in totally different tests, which looks
exactly like a real regression (a missing "done" event) until you notice the
test containers are just full of old fixtures. `pytest_sessionstart` below
wipes both disposable containers back to empty before every local run
specifically to make that class of flakiness impossible, not just to fix it
after the fact each time it shows up.
"""
import asyncio
import os
from pathlib import Path

if not os.environ.get("CI"):
    os.environ["DATABASE_URL"] = (
        "postgresql://orthopedics:orthopedics-dev@localhost:5433/orthopedics_agents_test"
    )
    os.environ["NEO4J_URI"] = "bolt://localhost:7688"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "orthopedics-dev"


async def _reset_postgres(database_url: str) -> None:
    """TRUNCATEs every table in the public schema (including LangGraph's own
    checkpoint tables, once AsyncPostgresSaver.setup() has created them on a
    prior run -- CASCADE handles the FK web between them without needing to
    know or maintain the exact table list here) except `alembic_version`
    (truncating it would make the very next `alembic upgrade head` in this
    same sessionstart re-run every migration from scratch, unnecessarily).

    app_settings is a singleton table (a check constraint pins it to exactly
    one row, id=1 -- see its own migration) that every real request assumes
    already has a row, so it's reseeded immediately after the wipe, mirroring
    the INSERT that migration itself runs on a fresh database.
    """
    import psycopg

    conn = await psycopg.AsyncConnection.connect(database_url)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                "AND tablename != 'alembic_version'"
            )
            tables = [row[0] for row in await cur.fetchall()]
            if tables:
                quoted = ", ".join(f'"{t}"' for t in tables)
                await cur.execute(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE")
            if "app_settings" in tables:
                await cur.execute(
                    "INSERT INTO app_settings (id, default_workflow) VALUES (1, 'deterministic')"
                )
        await conn.commit()
    finally:
        await conn.close()


async def _reset_neo4j(uri: str, user: str, password: str) -> None:
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    finally:
        await driver.close()


def pytest_sessionstart(session):
    """Bring the test database's schema up to date and the graph's
    constraints into existence before any test runs -- mirrors CI's separate
    `alembic upgrade head` step, but done here so a fresh (or just-reset)
    test container needs no manual migration step. Idempotent, so this is
    cheap even when the schema is already current. Wipes both local test
    containers first (see module docstring) -- CI doesn't need this, its
    containers are already fresh every run.
    """
    if not os.environ.get("CI"):
        asyncio.run(_reset_postgres(os.environ["DATABASE_URL"]))
        asyncio.run(
            _reset_neo4j(
                os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]
            )
        )

    from alembic import command
    from alembic.config import Config

    backend_root = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(backend_root / "alembic.ini"))
    command.upgrade(alembic_cfg, "head")

    from retrieval.graph_client import GraphClient

    async def _ensure_constraints() -> None:
        client = GraphClient(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"],
        )
        try:
            await client.ensure_constraints()
        finally:
            await client.close()

    asyncio.run(_ensure_constraints())

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


def pytest_sessionstart(session):
    """Bring the test database's schema up to date and the graph's
    constraints into existence before any test runs -- mirrors CI's separate
    `alembic upgrade head` step, but done here so a fresh (or just-reset)
    test container needs no manual migration step. Idempotent, so this is
    cheap even when the schema is already current.
    """
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

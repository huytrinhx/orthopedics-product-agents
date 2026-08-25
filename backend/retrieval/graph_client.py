"""Thin wrapper around the Neo4j driver, pointed at AuraDB by default (see
ARCHITECTURE decision: AuraDB now, self-hosted Neo4j-on-AKS is a config
change later — swap the URI/credentials, not this client).
"""
from neo4j import AsyncGraphDatabase


class GraphClient:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def query_related_entities(
        self, entity: str, relationship: str | None = None
    ) -> list[dict]:
        raise NotImplementedError

    async def get_synonym_groups(self) -> list[list[str]]:
        """Canonical synonym clusters, source of truth for query-time expansion
        via backend/agents/tools/synonym_resolve.py."""
        raise NotImplementedError

    async def close(self) -> None:
        await self._driver.close()

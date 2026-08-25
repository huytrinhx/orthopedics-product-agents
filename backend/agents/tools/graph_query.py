"""LangGraph tool wrapping Neo4j/AuraDB multi-hop entity queries.

See backend/retrieval/graph_client.py for the underlying client.
"""
from langchain_core.tools import tool


@tool
async def graph_query(entity: str, relationship: str | None = None) -> list[dict]:
    """Query the product/clinical entity graph for related entities."""
    raise NotImplementedError

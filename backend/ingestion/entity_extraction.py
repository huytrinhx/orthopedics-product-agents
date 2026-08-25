"""Extracts entities/relationships from chunks and writes them into the
AuraDB graph (backend/retrieval/graph_client.py) — the graph is populated
at ingestion time, not query time.
"""


async def extract_entities(chunk_text: str) -> list[dict]:
    raise NotImplementedError

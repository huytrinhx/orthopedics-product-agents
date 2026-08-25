"""Generates embeddings for chunks via OpenAI, used at ingestion time and
at query time (backend/agents/tools/vector_search.py).
"""


async def embed_texts(texts: list[str]) -> list[list[float]]:
    raise NotImplementedError

"""Generates embeddings for chunks via OpenAI, used at ingestion time and
at query time (backend/agents/tools/vector_search.py).
"""
import asyncio

from config.llm_clients import get_embeddings_model

# OpenAI's documented per-request item cap for the embeddings endpoint --
# batching at the real limit costs a small amount of retry work on a
# transient failure but saves far more in round-trips for a long document.
_BATCH_SIZE = 2048
_RETRY_DELAYS_SECONDS = (1, 2, 4)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embeddings_model()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        vectors.extend(await _embed_batch_with_retry(model, batch))
    return vectors


async def _embed_batch_with_retry(model, batch: list[str]) -> list[list[float]]:
    delays = (0, *_RETRY_DELAYS_SECONDS)
    for attempt, delay in enumerate(delays):
        if delay:
            await asyncio.sleep(delay)
        try:
            return await model.aembed_documents(batch)
        except Exception:
            if attempt == len(delays) - 1:
                raise
    raise AssertionError("unreachable")

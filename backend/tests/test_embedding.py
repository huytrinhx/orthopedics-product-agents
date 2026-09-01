"""Exercises backend/ingestion/embedding.py. Batching/retry control flow
around the OpenAI call is tested with a fake embeddings model (no network,
no key needed -- same "deterministic fake vectors" spirit as
test_vector_store.py); the real call itself needs OPENAI_API_KEY and is
skipped without it, same pattern as test_entity_extraction.py.
"""
import os

import pytest

from ingestion import embedding

needs_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="requires a real OPENAI_API_KEY"
)


class _FakeEmbeddingsModel:
    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.calls: list[list[str]] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("simulated transient failure")
        return [[float(len(text))] for text in texts]


async def test_embed_texts_returns_empty_for_no_input():
    assert await embedding.embed_texts([]) == []


async def test_embed_texts_batches_at_the_documented_cap(monkeypatch):
    fake = _FakeEmbeddingsModel()
    monkeypatch.setattr(embedding, "get_embeddings_model", lambda: fake)
    monkeypatch.setattr(embedding, "_BATCH_SIZE", 2)

    texts = ["a", "bb", "ccc", "dddd", "e"]
    vectors = await embedding.embed_texts(texts)

    assert [len(batch) for batch in fake.calls] == [2, 2, 1]
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [1.0]]


async def test_embed_texts_retries_transient_failures(monkeypatch):
    fake = _FakeEmbeddingsModel(fail_times=2)
    monkeypatch.setattr(embedding, "get_embeddings_model", lambda: fake)
    monkeypatch.setattr(embedding, "_RETRY_DELAYS_SECONDS", (0, 0))

    vectors = await embedding.embed_texts(["hello"])

    assert vectors == [[5.0]]
    assert len(fake.calls) == 3  # two failures, then a success


async def test_embed_texts_raises_after_exhausting_retries(monkeypatch):
    fake = _FakeEmbeddingsModel(fail_times=99)
    monkeypatch.setattr(embedding, "get_embeddings_model", lambda: fake)
    monkeypatch.setattr(embedding, "_RETRY_DELAYS_SECONDS", (0, 0))

    with pytest.raises(RuntimeError, match="simulated transient failure"):
        await embedding.embed_texts(["hello"])


@needs_openai_key
async def test_embed_texts_returns_real_1536_dim_vectors():
    [vector] = await embedding.embed_texts(["a guidewire is a thin wire used to guide implants"])
    assert len(vector) == 1536

"""Exercises backend/catalog/catalog_skus.py and its use by the deterministic
workflow's self_eval (issue #29) -- a fake graph, no Neo4j."""
import pytest

from agents.workflows import deterministic
from catalog import catalog_skus
from catalog.catalog_skus import load_catalog_skus


class _FakeGraphClient:
    def __init__(self, skus: list[str] | Exception):
        self.skus = skus
        self.calls = 0

    async def list_part_skus(self) -> list[str]:
        self.calls += 1
        if isinstance(self.skus, Exception):
            raise self.skus
        return self.skus


@pytest.fixture
def graph(monkeypatch):
    catalog_skus.clear_catalog_skus_cache()
    client = _FakeGraphClient(["SNAPWIRE", "MSK42020"])
    monkeypatch.setattr(catalog_skus, "get_graph_client", lambda: client)
    yield client
    catalog_skus.clear_catalog_skus_cache()


async def test_skus_are_cached(graph):
    assert await load_catalog_skus() == frozenset({"SNAPWIRE", "MSK42020"})
    assert await load_catalog_skus() == frozenset({"SNAPWIRE", "MSK42020"})
    assert graph.calls == 1


async def test_the_cache_expires(graph, monkeypatch):
    await load_catalog_skus()
    monkeypatch.setattr(catalog_skus, "CATALOG_SKUS_CACHE_SECONDS", 0)
    await load_catalog_skus()
    assert graph.calls == 2


async def test_an_unreachable_graph_answers_empty_and_isnt_cached(graph):
    graph.skus = OSError("connection refused")
    assert await load_catalog_skus() == frozenset()

    graph.skus = ["SNAPWIRE"]
    assert await load_catalog_skus() == frozenset({"SNAPWIRE"})
    assert graph.calls == 2


async def test_self_eval_checks_alpha_only_catalog_skus(graph):
    state = {
        "answer": "Place the SNAPWIRE first.",
        "reranked": [{"chunk_id": "doc#0", "text": "MSK42020 Full Thread screw"}],
        "resolved_parts": [],
    }
    assert (await deterministic.self_eval(state))["fact_check_issues"] == [
        "SNAPWIRE doesn't appear in any catalog record or passage retrieved this turn."
    ]

    state["reranked"] = [{"chunk_id": "doc#0", "text": "SNAPWIRE Ø1.1 x 50mm Guidewire"}]
    assert (await deterministic.self_eval(state))["fact_check_issues"] == []

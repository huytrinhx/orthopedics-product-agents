"""Exercises the LangGraph tools (backend/agents/tools/graph_query.py,
synonym_resolve.py) against real Neo4j data -- no mocking.
"""
import uuid

from agents.tools.graph_query import graph_query
from agents.tools.synonym_resolve import synonym_resolve
from retrieval.graph_client import get_graph_client


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def test_graph_query_returns_compatible_parts():
    client = get_graph_client()
    tray = _unique("TRAY")
    plate_sku = _unique("PLATE")
    screw_sku = _unique("SCREW")
    await client.upsert_tray(tray, None)
    await client.upsert_part(plate_sku, tray)
    await client.upsert_part(screw_sku, tray)
    await client.upsert_compatible_with(plate_sku, screw_sku)

    result = await graph_query.ainvoke({"entity": plate_sku, "relationship": "COMPATIBLE_WITH"})
    assert any(r["related_entity"] == screw_sku for r in result)


async def test_graph_query_unknown_entity_returns_empty():
    result = await graph_query.ainvoke({"entity": _unique("NOPE")})
    assert result == []


async def test_synonym_resolve_returns_other_cluster_members():
    client = get_graph_client()
    canonical = _unique("CANONICAL")
    variant_a = _unique("VARIANT-A")
    variant_b = _unique("VARIANT-B")
    await client.upsert_synonym_cluster(canonical, [variant_a, variant_b])

    resolved = await synonym_resolve.ainvoke({"term": variant_a})
    assert canonical in resolved
    assert variant_b in resolved
    assert variant_a not in resolved


async def test_synonym_resolve_is_case_insensitive():
    client = get_graph_client()
    canonical = _unique("CANONICAL")
    variant = _unique("VARIANT")
    await client.upsert_synonym_cluster(canonical, [variant])

    resolved = await synonym_resolve.ainvoke({"term": variant.upper()})
    assert canonical in resolved


async def test_synonym_resolve_unknown_term_returns_empty():
    resolved = await synonym_resolve.ainvoke({"term": _unique("NOPE")})
    assert resolved == []

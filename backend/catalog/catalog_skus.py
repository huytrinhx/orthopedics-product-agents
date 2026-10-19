"""Every catalog SKU, for agents/fact_check.py (issue #29).

The fact check's SKU pattern needs a digit and six characters, so it never
sees catalog SKUs like `SNAPWIRE`, `MSTJONES`, `MA830` or `0304-45`. It
also checks any token that exactly matches a catalog SKU, which needs the
whole catalog's SKUs, not just the turn's Parts: an answer that names a
real SKU the turn never retrieved is exactly what it should flag.

Reading every Part on every turn would be a heavy query for a set that only
changes when the catalog is re-seeded, so load_catalog_skus caches it in
process, the way catalog/family_descriptions.py caches its descriptions.
"""
import logging
import time

from retrieval.graph_client import get_graph_client

logger = logging.getLogger(__name__)

# A re-seed shows up within the hour, or at the next deploy.
CATALOG_SKUS_CACHE_SECONDS = 3600

_cached_skus: frozenset[str] | None = None
_cached_at = 0.0


async def load_catalog_skus() -> frozenset[str]:
    """Every Part SKU in the graph, cached in process. A graph failure is
    logged and answered with an empty set, uncached, so the next turn tries
    again: the fact check then only checks SKUs its pattern finds, as it
    did before this existed, rather than failing the turn.
    """
    global _cached_skus, _cached_at
    if _cached_skus is not None and time.monotonic() - _cached_at < CATALOG_SKUS_CACHE_SECONDS:
        return _cached_skus
    try:
        skus = await get_graph_client().list_part_skus()
    except Exception as error:  # noqa: BLE001 -- any graph failure degrades the same way
        logger.warning("catalog_skus: graph unavailable, checking pattern-shaped SKUs only: %s", error)
        return frozenset()
    _cached_skus = frozenset(skus)
    _cached_at = time.monotonic()
    return _cached_skus


def clear_catalog_skus_cache() -> None:
    global _cached_skus
    _cached_skus = None

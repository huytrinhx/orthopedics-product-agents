"""One-off cleanup after issue #28 ships, run by hand like a migration:

    python -m ingestion.cleanup_graph_facts

Swaps the Procedure constraint (name -> per-document key, via
ensure_constraints), then deletes what prose extraction wrote before
#28 and no re-index can remove: DIFFERENTIATES_FROM edges with no
`document_id`, and Procedure nodes keyed by a global name. Re-index the
documents afterwards to write their per-document edges and Procedures.
Idempotent: a second run deletes nothing.
"""
import asyncio

from retrieval.graph_client import get_graph_client


async def main() -> None:
    client = get_graph_client()
    await client.ensure_constraints()
    edges, procedures = await client.delete_unattributed_prose_facts()
    print(
        f"Deleted {edges} DIFFERENTIATES_FROM edges with no document_id and "
        f"{procedures} Procedure nodes with no per-document key"
    )


if __name__ == "__main__":
    asyncio.run(main())

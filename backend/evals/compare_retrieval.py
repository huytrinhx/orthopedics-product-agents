"""Ad-hoc CLI to compare what the vector leg (ticket 06,
backend/agents/tools/vector_search.py) and the graph leg (ticket 07,
backend/agents/tools/graph_query.py + synonym_resolve.py) each return for a
query, side by side -- useful now that both are real, ahead of ticket 08's
chat workflow deciding how to actually combine/rank them.

Not a formal eval (see backend/evals/harness.py for that, ticket 13, still
stubbed) and not a stand-in for it -- this doesn't score anything against
expected_citations, it just prints both legs' raw output for a human to
eyeball. There's no NL entity-extraction step anywhere in this repo yet
(that's the chat workflow's job) so the graph leg here either uses
explicit --graph-entity values you give it, or falls back to trying each
word in the query -- a rough diagnostic aid, not a real resolver.

Usage:
    python -m evals.compare_retrieval "What's the difference between the two 1.4mm wires in the MIS set?"
    python -m evals.compare_retrieval "compatible screws" --graph-entity MPPA100L --relationship COMPATIBLE_WITH
"""
import argparse
import asyncio
import os

from agents.tools.graph_query import graph_query
from agents.tools.synonym_resolve import synonym_resolve
from agents.tools.vector_search import vector_search

_RULE = "=" * 70
# graph_query's entity matching includes a CONTAINS check on Part
# descriptions -- common short words spuriously substring-match real
# descriptions (e.g. "for" hit "SLEEVE, PARALLEL WIRES FOR PEG PLATE").
# Excluded here so the no-entity fallback isn't drowned in that noise; this
# is a diagnostic-aid list, not a claim these words are never meaningful.
_STOPWORDS = {
    "what", "whats", "which", "who", "how", "why", "when", "where",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "for", "of", "in", "on", "at", "to", "and", "or",
    "with", "this", "that", "these", "those", "it", "its", "i", "you",
    "set", "use", "used", "using",
}


def _candidate_terms(query: str) -> list[str]:
    words = [w.strip(".,?!:;\"'()").lower() for w in query.split()]
    # dedupe, keep order, drop short/stopword noise
    seen: set[str] = set()
    terms = []
    for word in words:
        if len(word) > 2 and word not in _STOPWORDS and word not in seen:
            seen.add(word)
            terms.append(word)
    return terms


async def _print_vector_leg(query: str, top_k: int) -> None:
    print(_RULE)
    print(f"VECTOR SEARCH (top_k={top_k})")
    print(_RULE)
    if not os.environ.get("OPENAI_API_KEY"):
        # embed_texts (backend/ingestion/embedding.py) needs a real key to
        # embed the query -- same environment-config gap the ingestion
        # pipeline guards around, not something to crash this script over.
        print("  (skipped -- OPENAI_API_KEY not set, can't embed the query)")
        print()
        return
    chunks = await vector_search.ainvoke({"query": query, "top_k": top_k})
    if not chunks:
        print("  (no chunks -- nothing ingested that matches, or the index is empty)")
    for chunk in chunks:
        preview = chunk["content"][:160].replace("\n", " ")
        section = f" [{chunk['section_title']}]" if chunk["section_title"] else ""
        print(f"  score={chunk['score']:.4f}  {chunk['citation']}{section}")
        print(f"    {preview}...")
    print()


async def _print_synonym_leg(terms: list[str]) -> None:
    print(_RULE)
    print("SYNONYM RESOLUTION (per candidate term)")
    print(_RULE)
    any_hit = False
    for term in terms:
        synonyms = await synonym_resolve.ainvoke({"term": term})
        if synonyms:
            any_hit = True
            print(f"  {term!r} -> {synonyms}")
    if not any_hit:
        print("  (no synonym-graph hits for any candidate term)")
    print()


async def _print_graph_leg(entities: list[str], relationship: str | None, top_k: int) -> None:
    print(_RULE)
    print(f"GRAPH QUERY (relationship={relationship or 'any'})")
    print(_RULE)
    any_hit = False
    for entity in entities:
        related = await graph_query.ainvoke({"entity": entity, "relationship": relationship})
        if not related:
            continue
        any_hit = True
        print(f"  entity={entity!r}: {len(related)} related")
        for row in related[:top_k]:
            print(
                f"    [{row['relationship']}/{row['direction']}] "
                f"{row['related_entity']} ({', '.join(row['related_labels'])})"
            )
    if not any_hit:
        print(f"  (no matches for any of: {entities})")
    print()


async def compare(
    query: str, graph_entities: list[str], relationship: str | None, top_k: int
) -> None:
    print(f"Query: {query!r}\n")
    candidate_terms = _candidate_terms(query)

    await _print_vector_leg(query, top_k)
    await _print_synonym_leg(candidate_terms)
    await _print_graph_leg(graph_entities or candidate_terms, relationship, top_k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare vector-leg and graph-leg retrieval for one query.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query")
    parser.add_argument(
        "--graph-entity",
        action="append",
        default=[],
        dest="graph_entities",
        help="Explicit entity string to try against graph_query (repeatable). "
        "Defaults to trying each word in the query.",
    )
    parser.add_argument("--relationship", default=None, help="Restrict graph_query to one edge type")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(compare(args.query, args.graph_entities, args.relationship, args.top_k))


if __name__ == "__main__":
    main()

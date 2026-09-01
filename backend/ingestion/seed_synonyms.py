"""One-off deterministic seed script for backend/evals/synonyms-map.csv into
the Neo4j synonym/abbreviation graph (see backend/retrieval/graph_client.py's
schema docstring). Run manually — this is not part of per-document ingestion:

    python -m ingestion.seed_synonyms [path/to/synonyms-map.csv]

The CSV packs two independent tables into one sheet, offset by one row:

    col:      0            1               2  3       4   5   6   7
    row0:     Full term     Abbreviations      Synonyms
    row1:     Full thread   FT                 Term 1  T2  T3  T4  Notes
    row2:     Partial ...   PT                 interference screw  tenodesis screw  ...
    ...

Row 1's columns 0-1 are real data (the first abbreviation pair); its columns
3-7 are the *header* for the synonym-cluster table, not data — that table's
data starts at row 2. Within a synonym cluster, the first non-empty term is
treated as canonical (the sheet doesn't otherwise mark one); this is an
arbitrary but deterministic choice, not a claim that it's the "more correct"
term. A trailing Notes column (e.g. "caddy is NOT a synonym here") is stored
as a property on the CanonicalTerm node for human audit, not as a graph edge
— see CONTEXT.md / ticket 07's schema design note on why.
"""
import asyncio
import csv
import sys
from pathlib import Path

from retrieval.graph_client import GraphClient, get_graph_client

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "backend" / "evals" / "synonyms-map.csv"


def parse_synonyms_map(path: Path) -> tuple[list[tuple[str, str]], list[tuple[list[str], str]]]:
    """Returns (abbreviations, synonym_clusters).

    abbreviations: list of (full_term, abbreviation) pairs.
    synonym_clusters: list of (terms, notes) — terms has 2+ entries, the
    first is canonical.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    abbreviations = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        full_term, abbreviation = row[0].strip(), row[1].strip()
        if full_term and abbreviation:
            abbreviations.append((full_term, abbreviation))

    synonym_clusters = []
    for row in rows[2:]:
        padded = row + [""] * max(0, 8 - len(row))
        terms = [cell.strip() for cell in padded[3:7] if cell.strip()]
        notes = padded[7].strip()
        if len(terms) >= 2:
            synonym_clusters.append((terms, notes))

    return abbreviations, synonym_clusters


async def seed_synonyms(client: GraphClient, path: Path = DEFAULT_PATH) -> None:
    abbreviations, synonym_clusters = parse_synonyms_map(path)

    for full_term, abbreviation in abbreviations:
        await client.upsert_abbreviation(canonical=full_term, abbreviation=abbreviation)

    for terms, notes in synonym_clusters:
        canonical, *variants = terms
        await client.upsert_synonym_cluster(
            canonical=canonical, terms=variants, notes=notes or None
        )

    print(
        f"Seeded {len(abbreviations)} abbreviations and "
        f"{len(synonym_clusters)} synonym clusters from {path}"
    )


async def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    client = get_graph_client()
    await client.ensure_constraints()
    await seed_synonyms(client, path)


if __name__ == "__main__":
    asyncio.run(main())

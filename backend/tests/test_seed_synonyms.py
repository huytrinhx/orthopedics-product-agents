"""Pure parsing logic for backend/ingestion/seed_synonyms.py against the
real fixture (backend/evals/synonyms-map.csv) -- no Neo4j needed.
"""
from pathlib import Path

from ingestion.seed_synonyms import parse_synonyms_map

FIXTURE = Path(__file__).resolve().parents[1] / "evals" / "synonyms-map.csv"


def test_parses_abbreviations_not_the_synonym_sub_header():
    abbreviations, _ = parse_synonyms_map(FIXTURE)
    assert ("Full thread", "FT") in abbreviations
    # Row 1's columns 3-7 ("Term 1", "T2", ...) are the synonym table's
    # header, not data -- they must never show up as an abbreviation pair.
    assert not any(full == "Term 1" for full, _ in abbreviations)


def test_parses_synonym_clusters_starting_one_row_later_than_abbreviations():
    _, clusters = parse_synonyms_map(FIXTURE)
    cluster_terms = [terms for terms, _ in clusters]
    assert ["guidepin", "guidewire", "wire", "pin"] in cluster_terms
    # The synonym table's own header row must not appear as cluster data.
    assert not any(terms[0] == "Term 1" for terms in cluster_terms)


def test_exclusion_note_is_captured_not_dropped():
    _, clusters = parse_synonyms_map(FIXTURE)
    tray_cluster = next(terms for terms, _ in clusters if terms[0] == "tray")
    notes = next(notes for terms, notes in clusters if terms == tray_cluster)
    assert "caddy" in notes.lower()
    assert "caddy" not in tray_cluster

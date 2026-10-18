"""Deterministic answer checks run inline by every workflow's self_eval
node -- replaces the per-turn LLM-judge call there. agents/judge.py stays
for what it's actually good at (the offline eval harness and lining up
against human feedback scores); it just no longer runs on every chat turn.

Each check compares the draft answer against what the workflow actually
had in hand this turn -- the passages it retrieved and the catalog Part
records it resolved -- and reports a plain-language issue the model can fix
on a correction pass. No model call, so a clean answer costs nothing extra,
and a failure says exactly what's wrong instead of a 0-1 score.

Checks are deliberately conservative: each one only flags something it can
prove from the sources (a citation id that was never given, a SKU that
appears nowhere, a thread type or tray the catalog contradicts). Anything it can't
decide -- a line naming two parts with different thread types or trays, a part with
no thread property -- passes, so a false alarm never costs a correction
round.
"""
import re

from agents.citations import extract_citations, strip_citations

# SKU-shaped token: 2+ leading capitals, at least one digit, 6+ characters
# -- matches the catalog's shapes (MPPA100L, MSK42020, MSD14020) while
# skipping "MIS", "FT", "T8" and measurements like "4.0mm".
_SKU_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*\b")
_MIN_SKU_LENGTH = 6

# The abbreviations are case-sensitive on purpose ("ft" is also feet); the
# spelled-out forms aren't.
_THREAD_CLAIM_PATTERNS = {
    "FT": re.compile(r"\bFT\b|(?i:\bfull[- ]?thread)"),
    "PT": re.compile(r"\bPT\b|(?i:\bpartial(?:ly)?[- ]?thread)"),
}
_THREAD_LABELS = {"FT": "Full Thread (FT)", "PT": "Partial Thread (PT)"}


def _thread_category(thread: str | None) -> str | None:
    """Collapses the catalog's thread values ("Full", "Full, Cancellous",
    "Partial (1/3)", "Partial (16mm)", ...) to FT/PT; anything else
    ("Smooth", empty) has no FT/PT claim to check against."""
    lowered = (thread or "").strip().lower()
    if lowered.startswith("full"):
        return "FT"
    if lowered.startswith("partial"):
        return "PT"
    return None


def _check_citations(answer: str, citation_ids: set[str]) -> list[str]:
    return [
        f"Cited [{citation}], but no passage with that id was retrieved this turn."
        for citation in extract_citations(answer)
        if citation not in citation_ids
    ]


def _check_skus(answer: str, source_text: str, known_skus: set[str]) -> list[str]:
    issues: list[str] = []
    seen: set[str] = set()
    for sku in _SKU_PATTERN.findall(strip_citations(answer)):
        if len(sku) < _MIN_SKU_LENGTH or sku in seen:
            continue
        seen.add(sku)
        if sku not in known_skus and sku not in source_text:
            issues.append(f"{sku} doesn't appear in any catalog record or passage retrieved this turn.")
    return issues


def _check_thread_types(answer: str, parts: list[dict]) -> list[str]:
    thread_by_sku = {
        part["sku"]: part.get("thread")
        for part in parts
        if part.get("sku") and _thread_category(part.get("thread"))
    }
    issues: list[str] = []
    for line in strip_citations(answer).splitlines():
        skus_on_line = [sku for sku in thread_by_sku if re.search(rf"\b{re.escape(sku)}\b", line)]
        categories = {_thread_category(thread_by_sku[sku]) for sku in skus_on_line}
        # Two parts with different thread types on one line -- can't tell
        # which claim belongs to which, so don't guess.
        if len(categories) != 1:
            continue
        [actual] = categories
        claimed = {category for category, pattern in _THREAD_CLAIM_PATTERNS.items() if pattern.search(line)}
        if claimed and actual not in claimed:
            [wrong] = claimed
            for sku in skus_on_line:
                issues.append(
                    f"{sku} is {_THREAD_LABELS[actual]} in the catalog "
                    f"(thread: {thread_by_sku[sku]}), but the answer calls it {_THREAD_LABELS[wrong]}."
                )
    return issues


def _normalized_tray_text(text: str) -> str:
    # Tray names carry marks an answer often drops ("REFLEX® MINI"), so
    # both sides are compared without them, case- and spacing-insensitively.
    return " ".join(re.sub("[®™]", "", text).lower().split())


def _trays_named_on(line: str, tray_names: set[str]) -> set[str]:
    """The tray names written out in full on `line`. A name that's only
    part of a longer one also named there ("Foo Tray" inside "Foo Tray
    XL") isn't counted on its own."""
    normalized_line = _normalized_tray_text(line)
    named = {tray for tray in tray_names if _normalized_tray_text(tray) in normalized_line}
    return {
        tray
        for tray in named
        if not any(
            other != tray and _normalized_tray_text(tray) in _normalized_tray_text(other) for other in named
        )
    }


def _check_trays(answer: str, parts: list[dict], tray_names: set[str]) -> list[str]:
    """Issue #21: a tray named on the same line as a SKU has to be one the
    catalog puts that Part in. Only `tray_names` and the trays of this
    turn's own Parts are recognized as tray names, so a tray the turn never
    saw passes."""
    trays_by_sku = {part["sku"]: frozenset(part["trays"]) for part in parts if part.get("sku") and part.get("trays")}
    tray_names = set(tray_names).union(*trays_by_sku.values())
    issues: list[str] = []
    for line in strip_citations(answer).splitlines():
        skus_on_line = [sku for sku in trays_by_sku if re.search(rf"\b{re.escape(sku)}\b", line)]
        tray_sets = {trays_by_sku[sku] for sku in skus_on_line}
        # Parts from different trays on one line -- can't tell which tray
        # claim belongs to which, so don't guess.
        if len(tray_sets) != 1:
            continue
        [actual] = tray_sets
        named = _trays_named_on(line, tray_names)
        # Any right tray on the line passes: a "Conflicting sources" line
        # names the catalog's tray and the passage's wrong one together.
        if named and not named & actual:
            for sku in skus_on_line:
                issues.append(
                    f"{sku} is in {' / '.join(sorted(actual))} in the catalog, "
                    f"but the answer places it in {' / '.join(sorted(named))}."
                )
    return issues


def check_answer(
    answer: str,
    *,
    citation_ids: set[str],
    source_text: str,
    parts: list[dict],
    tray_names: set[str] = frozenset(),
) -> list[str]:
    """Every problem found in `answer`, as sentences addressed to the model
    that wrote it; an empty list means it passed.

    `citation_ids` are the `{document_id}#{chunk_index}` ids of every
    passage the model was shown, `source_text` is all of the text it was
    shown (passages and tool output alike), and `parts` are the catalog Part
    records it was given -- each a flat dict with at least `sku`, as
    graph_client.find_parts returns them (`trays`, when present, as
    part_lookup's display names). `tray_names` are any other tray display
    names to recognize in the answer -- e.g. every tray of the turn's
    family, so a draft that puts a Part in a sibling tray no resolved Part
    belongs to is still caught.
    """
    known_skus = {part["sku"] for part in parts if part.get("sku")}
    return [
        *_check_citations(answer, citation_ids),
        *_check_skus(answer, source_text, known_skus),
        *_check_thread_types(answer, parts),
        *_check_trays(answer, parts, tray_names),
    ]


def format_issues_for_correction(issues: list[str]) -> str:
    """The note a workflow hands back to the model on its one correction
    pass -- shared so both workflows phrase the retry the same way."""
    bullets = "\n".join(f"- {issue}" for issue in issues)
    return (
        "An automated check against the catalog and the retrieved passages found these "
        f"problems in your draft answer:\n{bullets}\n\n"
        "Rewrite the full answer with these fixed. Only state what the sources support; if a "
        "fact can't be confirmed, say so instead of guessing."
    )

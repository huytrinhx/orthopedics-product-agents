"""Shared citation-marker parsing for any workflow that asks the model to
cite context passages inline as `[doc-id#chunk-index]` (a real Postgres
`documents.id` UUID, not a SKU or other identifier -- see the pattern's own
comment). Promoted out of `agents/workflows/deterministic.py` so
`react_agent.py` doesn't carry its own copy of the same regex, including the
same UUID-validation fix: an unvalidated pattern let a bracketed non-UUID
identifier (e.g. a SKU cited the way ticket 22's catalog facts are meant to
be named inline, without brackets) reach `api/routes/chat.py`'s
`_resolve_citations`, which calls `uuid.UUID()` on it and crashed the whole
SSE stream ("badly formed hexadecimal UUID string") instead of just
dropping the one malformed citation.
"""
import re

_CITATION_REF = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}#\d+"
_SINGLE_CITATION_PATTERN = re.compile(_CITATION_REF)
# A bracket containing one *or more* comma-separated refs -- found live
# (2026-09-05) that the model sometimes cites two sources for one claim as
# "[id-a#3, id-b#5]" rather than two separate brackets. The original
# single-ref-per-bracket pattern left that whole bracket unmatched (no
# comma allowed inside), so extract_citations silently missed both
# references entirely -- not just a display glitch, the citation chip for
# either source never appeared at all. frontend/lib/chat/format.ts's
# stripCitationMarkers mirrors this exact pattern shape for the same reason.
_CITATION_GROUP_PATTERN = re.compile(
    rf"\[({_CITATION_REF}(?:\s*,\s*{_CITATION_REF})*)\]"
)


def extract_citations(answer: str) -> list[str]:
    """Every distinct `{document_id}#{chunk_index}` marker in `answer`, in
    first-seen order -- used to report only the citations the answer
    actually used, not every passage that happened to be in its context.
    """
    seen: dict[str, None] = {}
    for group_match in _CITATION_GROUP_PATTERN.finditer(answer):
        for single_match in _SINGLE_CITATION_PATTERN.finditer(group_match.group(1)):
            seen.setdefault(single_match.group(0), None)
    return list(seen)

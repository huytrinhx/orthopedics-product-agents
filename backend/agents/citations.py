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

_CITATION_PATTERN = re.compile(
    r"\[([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}#\d+)\]"
)


def extract_citations(answer: str) -> list[str]:
    """Every distinct `{document_id}#{chunk_index}` marker in `answer`, in
    first-seen order -- used to report only the citations the answer
    actually used, not every passage that happened to be in its context.
    """
    seen: dict[str, None] = {}
    for match in _CITATION_PATTERN.finditer(answer):
        seen.setdefault(match.group(1), None)
    return list(seen)

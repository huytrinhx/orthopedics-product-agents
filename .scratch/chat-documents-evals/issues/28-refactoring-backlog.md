# 28: Refactoring backlog -- not scheduled, revisit before the next big feature

Found during a 2026-09-05 deep-review pass (stale-docs audit + refactor scan)
across the full codebase, ~26 tickets in. None of these are bugs; all are
maintainability debt from features landing incrementally, one ticket at a
time, without a later pass to consolidate. Ordered by impact.

## 1. Repository connect/commit/close boilerplate

`backend/documents/repository.py`, `backend/feedback/repository.py`,
`backend/chat_threads/repository.py` -- every one of ~20 functions across
these three files repeats the identical shape:

```python
conn = await get_connection()
try:
    ...
    await conn.commit()  # where applicable
finally:
    await conn.close()
```

`config/db.py`'s `get_connection()` returns a `psycopg.AsyncConnection`,
which already supports `async with`. A small helper -- either
`async with get_connection() as conn:` at each call site, or a
`transaction()` async context manager in `config/db.py` that also commits
on clean exit -- would delete ~60 lines of copy-pasted ceremony and remove
the risk of a future function forgetting `.close()`. Mechanical, low-risk,
well covered by existing tests -- the best candidate to actually do first.

## 2. `frontend/app/chat/page.tsx` is a god component

521 lines, 17+ `useState` calls, owning thread-sidebar state/loading,
message send + SSE streaming, the clarification-interrupt flow (ticket 09),
and the citation pane (chunk fetch + PDF blob fetch + error states, ticket
26). Each ticket added its own state slice here rather than extracting one.
Suggested split: a `useChatStream` hook (send/stream/clarification state,
reusable independent of the page) plus a standalone `CitationPane`
component owning `docChunks`/`docFileData`/`docPaneError`/etc.

## 3. Chat streaming/clarification logic duplicated in the eval rerun view

`frontend/app/evals/rerun-conversation.tsx` independently re-implements the
same SSE-streaming + clarification state machine as `page.tsx` (both hand-roll
`messages`, `pendingClarification`, `error`, and the same
`STATUS_LABELS[evt.data.node] ?? evt.data.node` line verbatim). Real
duplication risk: a future SSE-handling bug fix (like the multi-citation
regex fix from this same session) has to be applied in two places, and it's
easy to fix one and miss the other. Should consume item 2's `useChatStream`
hook once it exists, rather than maintaining a parallel copy.

## 4. `backend/api/routes/chat.py` mixes four concerns in one 532-line module

SSE framing (`_sse`), citation resolution (`_resolve_citations`), the live
chat stream/resume endpoints, and the ticket-15 rerun endpoint (`rerun_chat`
plus its two request models, ~74 lines with its own checkpoint-seeding
logic). Lower priority than 1-3 -- it's already internally well-organized
with clear private-helper boundaries -- but `rerun_chat` would split
cleanly into its own `api/routes/chat_rerun.py` if this file keeps growing.

## Not flagged as issues (checked, found fine)

No dead code in `documents/repository.py` or `retrieval/graph_client.py`
(every top-level function has call sites). `tests/conftest.py`'s fixture
setup is already centralized, not duplicated. The `supervisor` workflow
stub (ticket 27) is intentionally kept registered-but-disabled per its own
ticket, not orphaned scaffolding.

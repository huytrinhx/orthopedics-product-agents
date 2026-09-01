# 18: Per-question observability (Langfuse tracing)

**What to build:** Every chat turn becomes one inspectable trace — latency
per step, tokens in/out per LLM call, every tool call (args/result/latency),
loop count, and the judge's final scores — all correlated to the literal
user question and inspectable in Langfuse's own UI, the same way in local
dev and production. Not scoped from the original 17-ticket list; requested
directly, designed via a grilling session (see
`.claude/plans/synthetic-spinning-bubble.md` for the full decision log)
before implementation.

**Blocked by:** 08 (Chat baseline workflow, end-to-end) — needs a real graph
to trace.

**Status:** done

## Design

### Mechanism: Langfuse's own LangChain/LangGraph callback handler, not manual instrumentation

`backend/observability/langfuse_setup.py`'s `new_callback_handler(*, user_id,
session_id, tags)` builds a `langfuse.langchain.CallbackHandler` (needs the
full `langchain` package, not just `langchain-core`/`langchain-openai` —
newly added to `pyproject.toml`) plus a config fragment carrying
`langfuse_session_id`/`langfuse_user_id`/`langfuse_tags` metadata.
`backend/api/routes/chat.py`'s `stream_chat` merges that into the `config`
dict it already passes to `graph.astream_events(...)` — one call site, zero
changes inside any workflow's node functions. Verified against a real trace:
17 automatic observations from one `deterministic` turn (every graph node,
every `ChatOpenAI` generation with real token in/out counts, every
`vector_search`/`synonym_resolve` tool call with latency), with no
special-casing per node.

### Trace = one chat turn, `session_id` = `thread_id`

Not one trace per whole conversation — each `/chat/{workflow}/stream` call
is its own trace, so "inspect this user's question" is always one trace,
not a span buried inside a long-running thread trace. `session_id =
thread_id` (already user-id-prefixed, see ticket 08) means Langfuse's own
session view still groups a conversation's traces together.

### Judge scores + loop count as trace-level scores, not per-retry

`score_trace(trace_id, *, eval_scores, loop_count)` fires exactly once,
after the graph run completes, using the *final delivered* answer's
`EvalScores` (faithfulness/relevance/style/citation) plus whichever
workflow-specific loop counter applies — five separate `NUMERIC` Langfuse
scores, independently filterable (e.g. "traces where citation < 0.6").
Internal retry-loop self-eval calls (`deterministic.py`'s `self_eval` node,
which can run more than once per turn) still show up as their own
generation spans automatically via the callback handler — this is
deliberately *not* duplicated as a score per retry, only the one that
corresponds to what the user actually received.

`_LOOP_COUNT_FIELD` (`chat.py`) maps `workflow_name -> state field name`
(today: `{"deterministic": "retrieval_loop_count"}`) rather than forcing
every workflow to rename its own counter to a common field — each
registered workflow's iteration semantics are architecturally different by
design (retrieval retry vs. open-ended tool-calling vs. delegation rounds),
so only the *field name at the observability call site* needs to be
consistent, not each workflow's internal logic. `react_agent`/`supervisor`
just report no `loop_count` until they're built and added to this map.

### Langfuse Cloud, not self-hosted

No PHI/compliance driver (confirmed), and volume (~50 sessions/day × 2-3
questions ≈ 100-150 traces/day, comfortably inside the free tier) makes the
self-hosted stack (web+worker+ClickHouse+Redis+MinIO) pure ops overhead for
no benefit. One Cloud project serves local dev, production, and offline
evals — `LANGFUSE_TRACING_ENVIRONMENT` (`local`/`production`/`eval`)
segments them instead of separate infra. README's step 4 updated to drop
the self-hosted `langfuse-local` clone-and-compose instructions.

### Graceful no-op when unconfigured

The Langfuse SDK itself logs a warning and returns a disabled client when
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are unset (verified directly),
rather than raising — so `configure_langfuse()` doesn't hard-fail at
startup, and CI (which sets neither) runs every non-`needs_openai_key` test
unaffected.

### `trace_url` in the SSE `done` event

`get_trace_url(trace_id)` is included alongside `eval_scores` in the `done`
payload, so a trace is one click away from the chat response itself, not
just findable by searching Langfuse's UI.

## Checklist

- [x] Every chat turn produces a real Langfuse trace: `session_id`,
      `user_id`, `tags`, `environment` all set correctly (verified via
      `client.api.trace.get(...)` against a real Langfuse Cloud trace, not
      just "the call didn't error")
- [x] Tokens in/out, latency, and every tool call are captured automatically
      per LLM/tool call, with zero code changes inside workflow node
      functions (verified: 17 real observations from one turn, including
      real token counts per `ChatOpenAI` generation)
- [x] Loop count and all four judge-score axes attached as trace-level
      scores, exactly once per turn, from the final answer only (verified:
      5 real scores read back from the trace)
- [x] `trace_url` returned in the chat SSE `done` event
- [x] Works identically local vs. production — same Langfuse Cloud project,
      only `LANGFUSE_TRACING_ENVIRONMENT` differs
- [x] Unconfigured Langfuse (no keys — e.g. CI) doesn't break the app;
      `backend/tests/` (100 tests) passes with real keys sourced from `.env`
- [x] `backend/tests/test_langfuse_setup.py` covers `score_trace`/
      `new_callback_handler`/`get_trace_url` with a mocked client (CI-safe,
      no real credentials or LLM call needed)

## Explicitly out of scope for this ticket

- **Generic OTel/FastAPI service-health instrumentation**
  (`backend/observability/otel_setup.py` stays a stub). Different audience
  (service health/error rates vs. per-question LLM detail) — separate,
  later work.
- **Offline eval harness tracing** (`backend/evals/harness.py`'s
  `run_eval`). Still `NotImplementedError` (ticket 13, blocked by 08, not
  yet started) — wire the same `new_callback_handler`/`score_trace` pattern
  in when that ticket implements `run_eval` for real, tagged
  `environment=eval` (see the module-level override note in
  `langfuse_setup.py`/`.env.example`).
- **Custom in-app trace dashboard.** Langfuse's own UI is the inspection
  surface — no page/panel built in this app for it.
- **Sampling / cost controls.** 100% capture for v1; add a `sample_rate`
  knob only if volume actually becomes a cost concern.
- **Local Postgres duplication of trace/score data.** Langfuse Cloud is the
  sole system of record; a future evals dashboard (ticket 15) should query
  Langfuse's API directly rather than this ticket pre-building a local copy.

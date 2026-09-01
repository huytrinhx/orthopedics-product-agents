# 08: Chat baseline (`deterministic`) workflow, end-to-end

**What to build:** A logged-in user can type a question in the chat UI and get back a real, cited, streamed answer. This wires up `backend/agents/workflows/deterministic.py`'s fixed pipeline (synonym_resolve → hybrid_retrieve → rerank → generate → self_eval, with a bounded retry loop) and `backend/api/routes/chat.py`'s streaming endpoint for real.

**Blocked by:** 02 (Email/password auth with admin flag), 06 (Ingestion vector leg), 07 (Ingestion graph leg)

**Status:** done

**Forward note (from ticket 07's schema design):** `graph_query` returns raw
facts with `SOURCED_FROM` document(s) (each carrying `doc_type`) but does
*not* rank them — applying `backend/evals/doctype-hierarchy.csv`'s
per-prompt-type source priority and its "confirm against ≥2 sources" rule
before answering belongs here (or a later ticket), since it also has to
reconcile against `vector_search` results from ticket 06, not just graph
facts alone. **Not done in this ticket** — see "Explicitly out of scope"
below; `graph_query` isn't called by this workflow at all (see "Design").

## Design

### `hybrid_retrieve` calls `vector_search` only, not `graph_query`

The module docstring's own pipeline name (`synonym_resolve → hybrid_retrieve
→ ...`) already answers what "hybrid" means here: `vector_search`
(`backend/agents/tools/vector_search.py`, ticket 06) is itself a hybrid of
vector + full-text search. `graph_query` isn't part of this fixed pipeline —
picking *which* entity to look up in the graph from a free-text question is
exactly the kind of judgment call a "fixed pipeline, no agentic tool choice"
baseline deliberately doesn't make. That's `react_agent`'s job (a later
ticket), where the model chooses which tools to call. `synonym_resolve`
still queries the graph internally, so this workflow does touch it, just
not `graph_query` directly.

### Graph shape

```
resolve_synonyms -> hybrid_retrieve -> rerank -> generate -> self_eval
                                                                  |
                                        scored low & retries left?
                                        yes -> reformulate -> resolve_synonyms (loop)
                                        no  -> finalize -> END
```

- **`resolve_synonyms`**: `synonym_resolve.ainvoke({"term": search_query})` —
  a single lookup against the whole query text, not per-extracted-entity.
  Usually a no-op for a full sentence; that's fine for a fixed baseline.
- **`hybrid_retrieve`**: `vector_search.ainvoke(...)` with `top_k=12` — a
  wider pool than what generation actually uses, so `rerank` has real
  candidates to choose among.
- **`rerank`**: one LLM call (structured output: a relevance score 0-1 per
  candidate, same order), not per-candidate calls. Narrows 12 candidates
  down to the top 5 by relevance rather than trusting embedding-similarity
  order alone.
- **`generate`**: answers only from the reranked passages, citing every
  claim as `[document_id#chunk_index]` (ticket 06's citation format). Also
  includes prior conversation turns (`state["messages"][:-1]`) so follow-ups
  work — see "Real bugs found" below for why that needed to be explicit.
- **`self_eval`**: `judge_answer` (new, `backend/agents/judge.py`) scores
  faithfulness/relevance/style/citation.
- **`reformulate`**: runs only when faithfulness or relevance scores below
  `RETRY_SCORE_THRESHOLD` (0.6) and the loop budget isn't spent
  (`MAX_RETRIEVAL_LOOPS = 2`). Rewrites `search_query` (not `query` — see
  below) and loops back to `resolve_synonyms`.
- **`finalize`**: the only node that appends to `messages`. See "Real bugs
  found."

### `query` vs. `search_query`

Two separate state fields, not one:
- `query` (from `BaseAgentState`): the user's actual question, verbatim,
  forever. `generate` echoes it back to the user and `self_eval` judges
  relevance against it — reformulate must never touch it.
- `search_query` (new, `DeterministicState`): what retrieval actually uses.
  Starts equal to `query`; `reformulate` is the only thing that rewrites it.

### Neither `search_query` nor `retrieval_loop_count` has a reducer

Both are last-write-wins fields with no `Annotated` reducer, which matters
for two different reasons documented where they're set:
- Within one turn, `reformulate`'s update is the only thing that changes
  them — the loop's internal bookkeeping.
- Across turns, the checkpointer restores whatever value a *previous* turn
  left them at, so `backend/api/routes/chat.py`'s `inputs` dict explicitly
  resets both (`search_query` to the new message, `retrieval_loop_count` to
  `0`) on every `/stream` call — seemingly redundant with what `resolve_synonyms`/entry
  already receives, but without it a prior turn's exhausted retry budget
  silently carries into the next turn. See
  `test_retrieval_loop_count_does_not_leak_into_a_fresh_turn` for the
  regression test.

### `judge_answer` (`backend/agents/judge.py`)

Structured-output LLM call (`_JudgeScores` pydantic model, one field per
axis, 0-1). Explicit instruction: an honest "the context doesn't answer
this" is *not* penalized on faithfulness just for being unhelpful — without
that, the self-eval retry loop would keep reformulating and retrying
forever on genuinely unanswerable questions instead of accepting an honest
non-answer.

### Checkpointer lifecycle (`backend/memory/checkpointer.py`, `backend/agents/registry.py`)

`AsyncPostgresSaver.from_conn_string()` is itself an async context manager
(the connection pool's lifetime is scoped to the `async with` block) — it
can't be a module-level singleton the registry builds lazily the way
`get_graph_client()`/`get_embeddings_model()` are. Resolved by:
- `memory.checkpointer.get_checkpointer(conn_string)`: an
  `asynccontextmanager` wrapping `AsyncPostgresSaver.from_conn_string()` +
  `.setup()` (idempotent — creates the checkpoint tables if missing).
- `backend/api/main.py` opens it once in a FastAPI `lifespan` and stores it
  on `app.state.checkpointer` — one checkpointer for the process's whole
  lifetime, the same pattern `graph_client.py`'s Neo4j driver singleton
  already established, not a fresh connection pool per chat request.
- `agents/registry.py`'s factory signature changed from `Callable[[],
  CompiledStateGraph]` to `Callable[[checkpointer], CompiledStateGraph]` —
  `react_agent.py`/`supervisor.py`'s stubs updated to match (trivial
  signature change, still `NotImplementedError`, not otherwise touched).

### SSE event schema (`backend/api/routes/chat.py`)

`POST /chat/{workflow}/stream` is a POST, so the native `EventSource` API
(GET-only) can't consume it — the frontend (`frontend/lib/api.ts`'s
`streamChat`) reads the fetch response body's stream and parses
`event: ...\ndata: ...\n\n` frames by hand. Event types, empirically
verified against `astream_events(version="v2")`'s actual shape (see
`git log` for the debug session that confirmed these, not guessed):
- `thread`: `{thread_id}` — first event, always, so the client can capture
  a freshly-generated thread_id for the next turn.
- `status`: `{node}` — emitted on `on_chain_start` where `event["name"] ==
  event["metadata"]["langgraph_node"]`, which is only true for a node's own
  top-level chain-start, not the LCEL sub-chains inside it (filtering on
  name-equals-node, rather than just "any chain-start with a node in
  metadata," is what excludes that noise).
- `token`: `{content}` — only from the `generate` node's
  `on_chat_model_stream` events; `rerank`/`self_eval`/`reformulate` also
  call the chat model but for structured (JSON) output the user isn't meant
  to watch arrive token by token.
- `done`: `{thread_id, answer, citations, eval_scores}` — from the root
  graph's own `on_chain_end` (`event["name"] == "LangGraph"`).
- `error`: `{message}` — any exception during the stream, reported over SSE
  rather than a bare mid-stream 500.

### Thread ownership

`thread_id` is server-generated as `f"{user_id}:{uuid4()}"` for a new
conversation. A client-supplied `thread_id` (resuming) is checked with a
prefix match before use; a mismatch is `403`. This is deliberately cheap
(no separate ownership table) and sets up ticket 10 (chat history sidebar)
to list a user's threads by the same prefix later.

### Citations are parsed from the answer, not dumped from the context window

`generate`'s prompt puts up to 5 reranked passages in context, but a given
answer usually only ends up citing a couple of them. `citations` in the
`done` event comes from regex-extracting `[doc-id#chunk-index]` markers out
of the actual answer text (`_extract_citations`), not from listing every
passage that happened to be available — showing citations for facts never
mentioned in the answer would overclaim sourcing.

## Real bugs found (during manual/browser testing, not caught by writing the code)

Each of these was caught by actually running the pipeline end-to-end
(`curl`, then a real browser) rather than by reasoning about the code, and
each now has a regression test:

1. **`generate` ignored all prior conversation turns.** It only ever built
   a prompt from the current turn's `query` + freshly retrieved context —
   `state["messages"]`'s earlier turns (which the checkpointer *was*
   correctly persisting) were never included in the LLM call. A follow-up
   like "what was my previous question?" got "I can't track conversation
   history" even though the history was sitting right there in state.
   Fixed by swapping the last message in `state["messages"]` for a
   context-augmented version and passing the rest through as real history.
2. **A retry appended two AI turns to permanent history, not one.**
   `generate` wrote to `messages` every time it ran, including the
   discarded first draft before a retry. Split into `generate` (produces a
   draft `answer`, doesn't touch `messages`) and a new `finalize` node
   (commits only the accepted answer to `messages`, runs once per turn on
   the accept path).
3. **`retrieval_loop_count` (and, relatedly, the reformulated query) could
   leak across turns via the checkpointer.** Neither field has a reducer,
   so a prior turn's checkpointed value would silently carry into a new
   turn unless explicitly reset. Fixed by having `chat.py` always pass
   `search_query`/`retrieval_loop_count` fresh in its `inputs` dict, and
   by separating `search_query` from `query` (see "Design" above) so a
   reformulated search string never leaks into what's shown to the user.
4. **`citations` in the `done` event included passages the answer never
   actually cited.** Fixed by extracting citations from the answer text
   itself instead of listing every reranked passage.

## Testing

Mirrors the established convention: real Postgres/Neo4j, no mocking, real
LLM calls gated behind an `OPENAI_API_KEY` `skipif`. One exception with its
own rationale: `test_deterministic_workflow.py`'s retry-loop tests
monkeypatch the node *functions* (`resolve_synonyms`, `generate`, etc.)
directly rather than faking the LLM client — `graph.add_node(name, fn)`
looks `fn` up from the module's globals at `build_graph()` call time, so
patching before building the graph swaps in deterministic fakes with no
LangChain/OpenAI mocking needed, which is what makes control-flow bugs like
#2 and #3 above testable at all without real API nondeterminism.

- [x] `POST /chat/{workflow_name}/stream` streams real `astream_events` output over SSE (node-by-node progress, not just a single final blob)
- [x] The `deterministic` workflow graph is built and compiled with the Postgres checkpointer, so a thread's state persists across turns
- [x] A real answer includes citations traceable to actual ingested chunks/documents
- [x] The self-eval retry loop (bounded, per `MAX_RETRIEVAL_LOOPS`) actually reformulates and retries when faithfulness/relevance score low, rather than being a no-op
- [x] The chat page renders the streamed answer live (progressive text/status), not just a spinner-then-final-answer
- [x] `thread_id`/`user_id` are tied to the logged-in user from ticket 02

## Explicitly out of scope for this ticket

- **`graph_query` integration.** Not called by this workflow — see
  "Design" above. `react_agent` (a later ticket) is where tool choice,
  including when to query the graph directly, belongs.
- **Doctype-priority ranking / cross-source confirmation**
  (`doctype-hierarchy.csv`'s P1-P4 ranking, "confirm against ≥2 sources").
  Still forward-noted from ticket 07, still not this ticket — there's only
  one retrieval leg (`vector_search`) actually wired into this workflow, so
  there's no second source to reconcile against yet.
- **Clarification via `interrupt()`/resume.** `POST
  /chat/{workflow_name}/resume` stays `NotImplementedError` — ticket 09.
- **Persisted history across page loads / a threads sidebar.** Ticket 10;
  this ticket's thread continuity is same-session only (React state), not
  reloaded from the checkpointer on page load.
- **Per-message 4-axis feedback UI.** Tickets 11/12.

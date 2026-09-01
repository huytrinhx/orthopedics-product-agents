# 08: Chat baseline (`deterministic`) workflow, end-to-end

**What to build:** A logged-in user can type a question in the chat UI and get back a real, cited, streamed answer. This wires up `backend/agents/workflows/deterministic.py`'s fixed pipeline (synonym_resolve → hybrid_retrieve → rerank → generate → self_eval, with a bounded retry loop) and `backend/api/routes/chat.py`'s streaming endpoint for real.

**Blocked by:** 02 (Email/password auth with admin flag), 06 (Ingestion vector leg), 07 (Ingestion graph leg)

**Status:** ready-for-agent

- [ ] `POST /chat/{workflow_name}/stream` streams real `astream_events` output over SSE (node-by-node progress, not just a single final blob)
- [ ] The `deterministic` workflow graph is built and compiled with the Postgres checkpointer, so a thread's state persists across turns
- [ ] A real answer includes citations traceable to actual ingested chunks/documents
- [ ] The self-eval retry loop (bounded, per `MAX_RETRIEVAL_LOOPS`) actually reformulates and retries when faithfulness/relevance score low, rather than being a no-op
- [ ] The chat page renders the streamed answer live (progressive text/status), not just a spinner-then-final-answer
- [ ] `thread_id`/`user_id` are tied to the logged-in user from ticket 02

**Forward note (from ticket 07's schema design):** `graph_query` returns raw
facts with `SOURCED_FROM` document(s) (each carrying `doc_type`) but does
*not* rank them — applying `backend/evals/doctype-hierarchy.csv`'s
per-prompt-type source priority and its "confirm against ≥2 sources" rule
before answering belongs here (or a later ticket), since it also has to
reconcile against `vector_search` results from ticket 06, not just graph
facts alone.

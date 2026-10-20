# agents.md — context for whoever (human or AI) works on this repo next

This file exists to carry forward decisions and working conventions so they
don't have to be re-derived or re-litigated. If you're an AI assistant
picking this repo up cold, read this before making structural changes.

## Standing technical decisions (don't silently reverse these)

- **Deployment is a single Railway service**, not AKS/Kubernetes. The root
  `Dockerfile` builds the Next.js static export and serves it plus the API
  from one FastAPI process (`backend/api/main.py` mounts `frontend/out`).
  This repo previously had Azure-native infra (Bicep: AKS, AI Search,
  Postgres Flexible Server, ACR) which was deliberately retired in favor of
  Railway — see `docs/adr/0001-railway-deployment.md`. Don't reintroduce
  Kubernetes/Bicep without discussing the tradeoff first.
- **Retrieval is Postgres/pgvector, not Azure AI Search.** The vector leg
  (`backend/retrieval/vector_store.py`) is pgvector cosine similarity; the
  keyword leg is Postgres full-text search (`tsvector`) — there is no
  AI-Search-style synonym-map index to keep in sync. Query-time synonym
  expansion instead queries the Neo4j/AuraDB graph directly
  (`backend/agents/tools/synonym_resolve.py`). See
  `docs/adr/0002-postgres-pgvector-retrieval.md`.
- **Chat goes through OpenRouter; embeddings stay on plain OpenAI**
  (`backend/config/llm_clients.py`), not Azure OpenAI. The enterprise
  service-principal auth pattern (`AZURE_TENANT_ID`/`CLIENT_ID`/
  `CLIENT_SECRET`) was removed along with Azure AI Search. See
  `docs/adr/0003-openai-direct-llm-provider.md` and
  `docs/adr/0007-openrouter-chat-gateway.md`, which supersedes 0003's chat
  half.
  - **OpenRouter is the only chat path once `OPENROUTER_API_KEY` is set.**
    `ChatOpenAI` stays the client with OpenRouter's `base_url`. Without the
    key (local dev, CI) chat goes to OpenAI direct. There's no per-call
    provider choice, and no provider-switch abstraction beyond that one
    branch: a new model, Kyma included, is an OpenRouter model id, not new
    client code. Every call site goes through `get_chat_model()`. Don't
    construct `ChatOpenAI` anywhere else, or that call skips the gateway
    and the fallback.
  - **`CHAT_MODEL` is an OpenRouter id and must be `openai/...`.** OpenAI
    direct runs it with the prefix stripped, and so does the outage
    fallback, which retries a 5xx, timeout, rejected key, out-of-credit
    402, or rate-limit/quota 429 against OpenAI direct and logs
    `chat_gateway_fallback`. Startup
    (`configured_chat_model()` in the lifespan) refuses anything else, and
    refuses a leftover `OPENAI_CHAT_MODEL` set on its own. Model output
    errors never trigger the fallback.
  - **`get_chat_model(model, reasoning_effort)` is `lru_cache`d per (model,
    effort).** `model` is an OpenRouter id (#24 picks one per node). The
    fallback always runs `CHAT_MODEL`.
  - **Every node uses the Responses API** (`USE_RESPONSES_API`), not Chat
    Completions. Production runs a GPT-5.x reasoning model, and OpenAI
    rejects function tools alongside reasoning on `/v1/chat/completions`,
    which broke every `react_agent` turn. Turning reasoning off
    (`reasoning_effort="none"`) was the other way out, but `react_agent`'s
    tool-bound `generate` call is where the model decides which tool to call
    and when to stop, so that's exactly where reasoning earns its cost.
    Whether OpenRouter's Chat Completions can replace it is
    `backend/evals/openrouter_spike.py`'s job, and ADR 0007 records the
    result. Responses-API messages carry `content` as a list of blocks: read
    model output with `.text`, never `.content`. OpenRouter's Responses API
    is stateless, which is why the gateway client sets `store=False`; don't
    turn on `use_previous_response_id`.
  - **Each node runs on its slot's model, and an unset slot means the
    default** (issue #24, `backend/model_assignments/`). There are six
    slots, each declaring what its LangChain call needs from a model:
    intent classification (`deterministic._classify_intent` and
    `_classify_question_type_reply`, structured output), deterministic
    answer (`deterministic.generate`, streamed text), ReAct agent
    (`react_agent.generate`, tool calling), judge (`agents/judge.py`,
    structured output), entity extraction and tray-layout extraction
    (structured output, plus image input for tray layout). An admin picks
    a model and reasoning effort per slot in Settings; it's a row in
    `model_assignments`. **A slot with no row runs `CHAT_MODEL` at the
    model's own reasoning default, by calling `get_chat_model()` with no
    arguments** — exactly what every node did before slots existed — so
    an empty table changes nothing. Without `OPENROUTER_API_KEY`, every
    slot runs the default whatever is saved, and Settings is read-only.
    - Slots are code (`slots.py`), like question types. A new call site
      joins an existing slot, or gets a new slot with its needs and probe
      checks (`probe.py`'s `SLOT_CHECKS`); don't call
      `get_chat_model(model, effort)` with a hand-picked model.
    - A chat turn resolves its slots once
      (`runtime.resolve_turn_slot_models`: saved settings through a
      30-second in-process cache, then the conversation's overrides), logs
      them as `chat_turn_models`, and passes them in the run config. Nodes
      read their slot with `slot_model_for_run`, never the database. The
      judge and each ingested document read theirs with
      `configured_slot_model`; ingestion logs `ingestion_models`. An
      extraction slot change only affects documents ingested afterwards.
    - Saving is two checks: the OpenRouter catalog's metadata has to cover
      the slot's needs (`catalog.py`, cached an hour), then the slot's real
      LangChain call has to work against the model on fixed sample inputs
      (`probe.py`, shared with `evals/openrouter_spike.py`). A failed probe
      rejects the save with the reason. There are no scheduled re-probes.
    - At runtime, an assigned model's unusable output (a parse or
      validation failure, an empty answer, a malformed tool call) is
      retried once on `CHAT_MODEL` and logged as `slot_output_fallback`
      (`runtime.invoke_with_output_fallback`). That's separate from the
      gateway-outage fallback, and a default slot never retries.
    - Changing the judge's slot breaks score comparison with earlier runs;
      Settings says so.
    - **Intent classification can also hold a decision model** (issue #23,
      `docs/adr/0008-decision-models-for-intent.md`): a second kind of
      option in the same slot (`model_kind` "decision"), not a separate
      config path. It runs on OpenRouter's Decisions API with the same
      `OPENROUTER_API_KEY`, through `get_decision_client()` beside
      `get_chat_model()`; don't call the Decisions API anywhere else.
      `agents/intent_decisions.py` asks one Choice per decision (family
      with `unclear`/`no_family` options, question type with a top-two
      margin rule, the question-type reply) and returns the chat path's
      shape, so the pause conditions stay the single place the decisions
      live. The family confidence threshold and question-type margin are
      saved with the slot (defaults 0.6 / 0.15 in `slots.py`, untuned).
      Only slots with `accepts_decision_models` offer one; decision ids
      are saved as their dated `canonical_slug`. Its own probe
      (`DECISION_SLOT_CHECKS`) must get valid option keys back. Any
      decision failure logs `slot_output_fallback` and runs the slot's
      default chat call.
    - **Both intent classifiers see a description of each family, not
      just its name** (issue #30). `catalog/family_descriptions.py`
      generates it from the family's trays and Parts (cached an hour). An
      admin can replace it in Settings → Families & trays; that's
      `product_families.description_override`, which a re-seed keeps.
      `load_family_descriptions` lays the overrides over the generated
      text, so an edit applies on the next turn. Read descriptions through
      it, never `describe_families` directly. A description edit changes
      what both classifiers see, so it's worth an intent eval run.
    - The Eval tab's "Rerun with models" sends admin-only per-slot
      overrides with the chat request (`model_overrides`; a non-admin gets
      a 403). They're kept in that conversation's state only, and its
      answers show what they ran on. The harness takes the same choice as
      `--model <slot>=<model_id>[@<effort>]`; an id OpenRouter lists as
      a decision model runs as one, and `--family-confidence-threshold` /
      `--question-type-margin` tune its thresholds for that run.
- **Uploaded/ingested documents live on local disk (`INGEST_DATA_DIR`), not
  object storage.** No Azure Blob Storage/Azurite — a Railway volume covers
  persistence in production. See
  `docs/adr/0005-local-disk-document-storage.md`.
- **Neo4j is AuraDB in production, Docker locally** — this is independent
  of the Railway/AKS decision above; AuraDB is Neo4j's own managed control
  plane either way, not something provisioned through app infra.
  - **Graph writes go through `GraphClient._write` /
    `_write_transaction`** (managed `execute_write` transactions), never a
    bare `session.run`: only a managed transaction is retried when Aura
    drops a pooled connection, so an auto-commit write fails the request
    instead (seen in production as a 503 on a tag save). The driver's
    `liveness_check_timeout`/`max_connection_lifetime` settings in
    `GraphClient.__init__` exist for the same reason; keep them. A retried
    transaction re-runs every statement, so writes stay MERGE/MATCH-based.
- **Judge rubric is shared between offline evals and human feedback.**
  `backend/agents/judge.py` scores faithfulness/relevance/style/citation for
  the offline eval harness (`backend/evals/harness.py`) and the human
  feedback UI — on the same schema, so human and automated scores stay
  directly comparable. Don't fork the rubric per call site.
  - **The harness has two evals** (issue #25), both run from `backend/`
    with `OPENAI_API_KEY`, both a CLI report (plus `--output` JSON), with
    nothing persisted.
    - `python -m evals.harness intent` (Slice 1, needs a seeded catalog
      mirror) scores family accuracy, question-type accuracy,
      clarification precision/recall/rate, latency and tokens on
      `intent_detection.jsonl`. It calls the deterministic workflow's own
      `_classify_intent`, `_resolve_family` and pause conditions
      (`_needs_family_clarification`, `_needs_question_type_clarification`)
      rather than copies, so keep those the single place the decisions live.
    - `python -m evals.harness answers --workflow <name>` (Slice 2,
      `evals/answer_eval.py`, needs ingested documents, the catalog mirror
      and Neo4j) runs every per-family QA row through a registered workflow
      as one fresh thread on an in-memory checkpointer, so it writes no
      checkpoints or chat threads. It reports the judge's per-axis
      averages (with the row's `expected_answer` passed as the judge's
      `reference_answer`), the workflow's own fact-check issue rate,
      citation recall (an expected citation counts when a document the
      answer cited carries every word of it, `names_document`), pause rate
      by kind, p50/p95 latency, tokens per model, and a table of failures
      and low scores.
    - **Pause policy (b): a paused row resumes with a scripted reply,**
      chosen by the node that paused: the row's `product_family` for the
      family question, its first `question_type` (as the offered option
      text) for the question-type one, and the first option for the
      synonym one (taken from `synonym_ambiguity`, since that interrupt
      offers none). A row's `clarification_replies` overrides these; it's
      written by `build_dataset.py`'s `CLARIFICATION_REPLY_OVERRIDES`, not
      by hand. At most 3 resumes per row (`MAX_RESUMES_PER_ROW`); a row
      still paused after that is a failure. If you add an interrupt() to
      a workflow, give its node a pause kind in `answer_eval.py`, or it
      gets its first option.
  - **Evals are run by an admin, by hand — never automatically in CI.**
    They cost real model calls. **Whenever a change touches the agent
    backend** (`backend/agents/`, or anything a workflow node calls into,
    such as `retrieval/` or `config/llm_clients.py`), whoever makes it —
    human or AI agent — tells the admin before it's merged or deployed
    that the evals should be run, and gives the command. Evals run
    locally only. There's no CI job for them, not even a manual one: a
    real answer eval needs a database with ingested documents, and the
    production credentials that implies stay out of GitHub.
- **Earlier turns reach the model as thread memory, not as `messages`**
  (issue #7, Ticket A; deterministic workflow only). `finalize` appends
  one small record per turn to `turn_memory` (append reducer, built from
  state with no model call), and `generate` sends one "Earlier in this
  conversation" block rendered from it (`memory/thread_memory.py`), capped
  at ~1,500 tokens by `memory/budget.py`, oldest dropped first. `messages`
  is the UI's transcript only; don't put it back into the prompt. The
  previous turn's SKUs are looked up again in `resolve_skus`, so the fact
  check has their catalog records. Each block's token count is a Langfuse
  score (`thread_memory_tokens`). Family carry-over (#22) is unchanged and
  separate.
- **Inline self-eval is a deterministic fact check, not the judge.** Every
  workflow's `self_eval` node runs `backend/agents/fact_check.py` — citation
  ids, SKUs, thread types, and trays checked against the turn's own retrieved
  passages and catalog Part records, no model call — and a failed check
  earns one correction pass (`MAX_CORRECTION_ROUNDS`) with the specific
  problems listed. This replaced a per-turn LLM-judge call: it costs nothing
  on a clean answer and says exactly what's wrong on a bad one. It's a
  separate signal from the judge rubric, not a fork of it; extend it with
  new checks that can be proven from the sources, and keep anything that
  needs judgment (style, relevance) in the judge.
- **A document request is answered with the document, not a summary of
  it** (deterministic workflow). A turn resolved to `documents_lookup` goes
  to `lookup_document`, which matches the request against indexed
  documents' filename, document type, family and tray tags
  (`agents/tools/document_lookup.py`'s `single_best_match`, within the
  turn's family). One clear winner answers with a fixed sentence citing
  it, with no retrieval, model call or fact check; a tie or a thin match
  carries on through the normal pipeline. Keep it a match in code, not a
  model call: a rep asking for a document should get it with no LLM in the way.
- **Workflow registry pattern.** Each agent architecture is its own
  LangGraph graph registered in `backend/agents/registry.py` under a shared
  state schema (`agents/state.py`) — the API and eval harness select by
  name. New architectures plug in via `agents/workflows/__init__.py`; they
  shouldn't require changes to callers.
- **Domain package pattern.** A domain's logic lives in its own package —
  `models.py` + `repository.py` (+ `service.py` where there's real
  processing, see `backend/documents/`) — with exactly one seam: the route
  file that imports from it (`backend/api/routes/chat.py` importing
  `chat_threads.models`/`chat_threads.repository`, `documents.py` importing
  `documents.*` and `tags.models`, etc.). No route file inlines model or
  persistence logic, and packages don't cross-import except where one
  domain genuinely depends on another (documents on tags, for tagging).
  This is why ticket 10 (chat history sidebar) needed zero `main.py`
  edits — `chat_threads/` is a self-contained package a ticket can own
  end-to-end. New domains follow this shape by default; don't grow logic
  directly inside `backend/api/routes/*.py`.
  - The frontend mirrors this 1:1 under `frontend/lib/`: one folder per
    domain (`auth/`, `documents/`, `chat/`), each holding `api.ts` +
    `types.ts` (+ `token.ts` for auth's session storage). `tags` nests
    under `documents/tags/` there — tags only exist to label documents,
    unlike the backend's dedicated `tags.py` admin route — so the two
    sides deliberately diverge in shape while sharing the same packaging
    convention. (The Settings page, `frontend/app/admin/`, imports it too:
    Document Types are a closed list curated only there, issue #16.) The one exception
    is `frontend/lib/api/client.ts`: shared HTTP plumbing (`API_BASE`,
    `authHeaders`, `request`, `unwrap`) that every domain imports — not a
    domain itself, low-churn, so it isn't a collision risk the way a flat
    `lib/api.ts` with every domain's functions appended to it was.
- **Question types are code; their doc-type hierarchy is data.** Slugs
  and descriptions live in `backend/agents/question_types.py`, because the
  deterministic workflow branches on them and the classifier's enum is
  built from them. Which document types each one retrieves from, and in
  what order, lives in the `question_type_document_types` table
  (`backend/doc_type_hierarchy/`, route `api/routes/question_types.py`).
  Admins edit it from Settings. `hybrid_retrieve`'s hard filter and
  `rerank`'s bonus read it by document type **id**. Don't bring back
  name/substring matching or an in-code priority dict. `unclassified` is
  always present, is read-only, and never has rows (there's a check
  constraint); like any question type with an empty list, it retrieves
  unfiltered. A doc-type merge moves the merged type's rows to the
  surviving type (`tags.repository.merge_document_type`). If you add a
  table that references `document_types`, make merge handle it too. See
  `docs/adr/0006-doc-type-hierarchy-is-data.md`.
- **Golden datasets are per Product Family, not generic.** There's one QA
  file per family a reviewed question is about
  (`mis-foot-recon-system.jsonl`, `reflex-foot-recon-systems.jsonl`,
  `reflex-hybrid.jsonl`), and `intent_detection.jsonl` labels each query
  with its expected family, question-type slugs, and whether it should
  trigger the family (#18) or question-type (#19) clarification. Rows
  with several `turns` label the last one as a follow-up to the earlier
  ones (#22, family carry-over).
  `feedback-notes.csv` keeps reviewers' original MIS/REFLEX column;
  `backend/evals/golden_datasets/build_dataset.py` maps it to families
  (REFLEX questions naming HYBRID go to REFLEX HYBRID) and raises on any
  value it has no mapping for. Rerun it after new rows land in the CSV,
  and don't hand-edit the JSONL. The one hand-written file is
  `technique_steps/step_skus.jsonl` (issue #28), the step -> SKU sets
  from the 2026-10-18 evaluation; it sits in a subdirectory so the
  answer eval's directory glob doesn't read it as QA rows.
- **Technique steps are extracted as an outline and resolved to SKUs in
  code** (issue #28, `backend/techniques/`). One model call per document,
  on the entity-extraction slot, outlines techniques and steps from the
  step-bearing vector chunks (`chunk_index` kept, so every step cites a
  real `{document_id}#{chunk_index}`). The model names instruments and
  specs ("Ø2.6mm drill for Ø3.7mm screws"), never a SKU it wasn't shown
  printed; `techniques/resolution.py` turns each item into a SKU set from
  the catalog's structured specs and the document's SKU index, bound to
  the parameter values it's for (`applies_when`). Don't move resolution
  into the prompt. Storage is Postgres (`techniques`, `technique_steps`,
  `technique_step_items`), replaced as a whole on re-index; the graph only
  gets one `Procedure` per document and technique (keyed
  `<document_id>::<name>`) with REQUIRES edges to its trays, and every
  `DIFFERENTIATES_FROM` edge carries its `document_id`. Read steps through
  `techniques.service.technique_steps(family, technique, parameters)`,
  which `python -m evals.harness steps` scores too.
- **Auth is JWT-based, stateless, client-held.** `backend/auth/` issues a
  signed JWT on signup/login (`JWT_SECRET`); the frontend stores it in
  `localStorage` (`frontend/lib/auth/token.ts`) and attaches it as a Bearer
  token — there's no server-side session store to invalidate on logout.
  `is_admin` is granted from the `ADMIN_EMAILS` allowlist at signup and
  re-checked on every later login (`auth.repository.promote_to_admin`, called
  from `/auth/login` and `/auth/google/callback`) — so adding someone to the
  allowlist after they already have an account promotes them on their next
  sign-in, no DB edit needed. It's still not editable via any UI. Admin-gated
  pages (Documents, Evals, Users) check this flag; regular chat access
  doesn't require it, but does require `is_active` (the Users tab's
  Enable/Disable) — a new non-admin signup starts `is_active=False` and is
  blocked from chat (`auth.dependencies.require_chat_access`) until an admin
  enables them. Admins bypass the `is_active` check entirely, so an admin's
  row is read-only in the Users tab. The Users tab also shows `last_login_at`
  per account, which in practice means "last seen": it's stamped on signup,
  every login/OAuth callback, *and* any authenticated request
  (`auth.dependencies.get_current_user` → `repository.touch_last_seen`),
  throttled to one write per 5 minutes (`LAST_SEEN_THROTTLE`). Without that,
  a user reusing a 30-day token would never move it.
- **Schema changes go through Alembic (`backend/migrations/`), not hand-run
  SQL.** `alembic upgrade head` reads `DATABASE_URL` (same env var as
  everything else) and applies pending migrations; it's a required step in
  local setup (README), CI (runs against a throwaway Postgres service before
  `pytest`), and production. Production is **manual**, not part of the
  deploy: there's no release/pre-deploy command, so after a deploy carrying
  a new migration goes live, someone runs `alembic -c alembic.ini upgrade
  head` from `backend/` with production's env (e.g. via `railway run`, or
  `railway ssh` where the network allows it — see README's "Deploying to
  Railway"). Until then the new code is serving against the old schema.
  Add new tables via `alembic revision -m "..."`, not a one-off script.

## Deployment direction (as of this writing)

Single Railway service, no GitHub Actions deploy step — `.railway/railway.ts`
(Railway's Infrastructure as Code; there is no `railway.toml`) points
Railway at the root `Dockerfile`, and it redeploys on every push to `main`.
Migrations and graph seeding are not part of that deploy; they're run by
hand afterwards (see the Alembic bullet above). Postgres is Supabase
(pgvector-enabled); Neo4j is AuraDB. Full step-by-step is in `README.md`'s
"Deploying to Railway" section. Highlights:

- The frontend is a Next.js **static export** (`output: "export"` in
  `frontend/next.config.js`) served by the backend — this rules out
  server-only Next.js features (route handlers, server actions, ISR). If a
  future feature genuinely needs one of those, that's a call to revisit the
  single-service topology (`docs/adr/0004-single-service-topology.md`), not
  to quietly bolt on a second Node process.
- `backend/api/main.py` mounts `frontend/out` as static files, registered
  **last** so it never shadows `/chat`, `/documents`, `/feedback` — Starlette
  matches routes in registration order.
- The Docker image installs the backend with `pip install -e ./backend`
  (editable), not a normal wheel install — `backend/api/main.py` locates
  `frontend/out` via a `__file__`-relative path, which only resolves
  correctly if the source tree stays where it was copied (`/app/backend`),
  not copied again into `site-packages`.

## Where to look for more

- `product.md` — what's actually known about the product surface, and
  what's still open.
- `build-log.md` — chronological record of how this repo got here.
- `CONTEXT.md` / `docs/adr/` — domain vocabulary and architectural decision
  records; see `docs/agents/domain.md` for how to consume them.
- `docs/agents/issue-tracker.md` — how issues are tracked for this repo.
- `README.md` — how to actually run the thing.

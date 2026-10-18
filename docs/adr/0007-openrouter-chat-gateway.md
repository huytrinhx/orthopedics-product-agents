# 0007 — Chat goes through OpenRouter, with OpenAI direct as fallback

## Status

Accepted. Supersedes the chat-provider half of
[0003](0003-openai-direct-llm-provider.md); 0003's embeddings decision
still stands. The API spike below is **pending**: it hasn't been run
against the production model yet.

## Context

ADR 0003 put both chat and embeddings on the OpenAI API directly. Swapping
the chat model then meant changing provider-specific client code.
Production moved from `gpt-4o` to a GPT-5.x reasoning model
(`gpt-5.6-luna`), and every tool-calling turn started failing with a 400,
because OpenAI rejects function tools alongside reasoning on
`/v1/chat/completions`. The fix was `use_responses_api=True` in
`get_chat_model()` and reading output with `.text`. Each new model or
provider can bring another quirk like that, and we want trying one to be a
config change. Issue #8 has the full discussion. Kyma, which 0003 left as an
open option, isn't adopted. OpenRouter covers it as one more model id if it
ever matters.

## Decision

- **OpenRouter is the only chat path once it's configured.** When
  `OPENROUTER_API_KEY` is set, every chat call goes through OpenRouter's
  OpenAI-compatible API (`https://openrouter.ai/api/v1`). `ChatOpenAI` is
  still the client, with OpenRouter's `base_url` and key. Without the key
  (local dev, CI), chat goes to OpenAI direct as before. There's no
  per-call choice of provider.
- **One default model, `CHAT_MODEL`, written as an OpenRouter id** such as
  `openai/gpt-5.6-luna`. The OpenAI-direct path strips the `openai/`
  prefix. It replaces `OPENAI_CHAT_MODEL`.
- **Startup validates `CHAT_MODEL`** (`configured_chat_model()`, called
  from the FastAPI lifespan). It must be an `openai/...` id, so the
  OpenAI-direct path, and with it the outage fallback, always has a model it
  can run. Startup also fails if `OPENAI_CHAT_MODEL` is still set without
  `CHAT_MODEL`, so a deploy carrying the old variable doesn't silently drop
  to the default model.
- **A gateway outage falls back to OpenAI direct.** A call that fails with a
  5xx, a timeout or dropped connection, or a rejected key is retried once
  against OpenAI direct with `CHAT_MODEL`, and a `chat_gateway_fallback` log
  line records it. A 4xx about the request, a rate limit, or model output
  that doesn't parse would fail the same way on OpenAI direct, so those
  still raise. `OPENAI_API_KEY` is always present, because embeddings use
  it.
- **Every node uses the same API, and for now it's the Responses API.**
  That's what works in production today. OpenRouter's Responses API is beta
  and stateless: it rejects `store: true` and `previous_response_id`.
  `ChatOpenAI` sends neither by default, but the gateway client sets
  `store=False` explicitly. Only then does langchain-openai stop replaying
  earlier responses' reasoning and message item ids, which a stateless
  server can't resolve, on `react_agent`'s second and later tool rounds.
  The fallback client is stateless too, because a turn that falls back
  partway through replays history the gateway produced. Plain OpenAI direct
  without a gateway keeps the server's default storage, as before.
- **`get_chat_model(model, reasoning_effort)`** takes an OpenRouter model id
  and an optional reasoning effort (both for #24's per-node models), and
  stays `lru_cache`d per (model, effort) pair. The fallback always runs
  `CHAT_MODEL`, because a per-call model may not exist on OpenAI direct.
- **Embeddings stay on OpenAI direct.** The `text-embedding-3-small`
  vectors already in pgvector would all need re-embedding otherwise.

## Spike: Responses API vs Chat Completions through OpenRouter

**Result: pending.** `backend/evals/openrouter_spike.py` sends every node's
call shape to OpenRouter under both APIs, using the production client
settings, and prints a pass/fail table. The call shapes are
`with_structured_output` with the real schemas for `detect_intent`, the
question-type reply classifier, the judge, entity extraction, and
tray-layout extraction (with image input); `bind_tools` with
`react_agent`'s tools over two rounds; and streamed text for deterministic
`generate`. Run it with `OPENROUTER_API_KEY` and the production
`CHAT_MODEL` set (`python -m evals.openrouter_spike` from `backend/`),
then record the table here.

- If Chat Completions passes every check with reasoning on, set
  `USE_RESPONSES_API = False` in `backend/config/llm_clients.py` and drop
  the Responses-API-only handling.
- If only the Responses API passes everything, nothing changes.
- Only if neither passes every node does the API become a per-model flag.

## Consequences

- Trying a different chat model is a `CHAT_MODEL` change, as long as it's
  an OpenAI model. Non-OpenAI models are #24's per-node choice, and they
  can't be the default, because the fallback couldn't run them.
- Production needs `OPENROUTER_API_KEY` and `CHAT_MODEL` set, with
  `OPENAI_CHAT_MODEL` removed (README, `.railway/railway.ts`).
- An OpenRouter outage costs latency (OpenAI client retries, then the
  fallback call), not answers. It shows up as `chat_gateway_fallback` log
  lines.
- On a `react_agent` turn through the gateway, the model's reasoning from
  one tool round isn't carried into the next, because the stateless replay
  drops reasoning items without `encrypted_content`. If the spike or answer
  quality shows that matters, request `reasoning.encrypted_content` through
  `include`.
- No batch eval confirms the gateway leaves scores unchanged, because
  `backend/evals/harness.py` only scores intent so far (#25 Slice 1). A
  manual Eval-tab comparison covers answers for now. The CI `evals` job,
  which passed without evaluating anything, is deleted; evals are run by
  an admin by hand (agents.md), not per PR.

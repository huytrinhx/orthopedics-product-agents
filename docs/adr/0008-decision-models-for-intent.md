# 0008 — Intent classification can run on a decision model, through OpenRouter

## Status

Accepted for the code path (issue #23). Not yet chosen for production: that
waits on the intent-eval comparison and the vendor review, both left to the
admin. Builds on [0007](0007-openrouter-chat-gateway.md) and #24's model
slots.

## Context

The deterministic workflow's classification steps (`_classify_intent`,
`_classify_question_type_reply`) choose from fixed lists, but they run on
the main chat model, a GPT-5.x reasoning model in production, through
`with_structured_output`. That's the slowest, most expensive way to pick
from a list, and it has no calibrated confidence: "prefer null over
guessing" is a prompt instruction, not a threshold we can tune.

Decision ("System One") models write no text. Given a state and typed
questions (Choice, Noul, Score), they answer in one forward pass with a
probability per option. Jev (TypeSafe), Microsoft Decision-1 and others
exist. The issue asked whether OpenRouter carries them, and to use
TypeSafe's own API otherwise.

OpenRouter does carry them. Its Decisions API is `POST
https://openrouter.ai/api/alpha/decisions`, authenticated with the same
`OPENROUTER_API_KEY` as chat. The body is `{model, state, questions}`, and
each Choice answer carries `choice`, `probabilities` and `confidence`;
the response also names the exact build that answered and reports `usage`
(tokens and cost). Decision models are listed at `GET
/api/v1/models?output_modalities=decisions`, each with a `canonical_slug`:
a dated build such as `typesafe/jev-1.13-20260917`. Both Jev and
Decision-1 are listed there. The API is labelled alpha.

## Decision

- **Provider: OpenRouter's Decisions API.** No new vendor account, key or
  env var: production already has `OPENROUTER_API_KEY`, and local dev and
  CI need nothing new, because an unset slot never calls it. Jev versus
  Decision-1 is a model id chosen in Settings, not a code change. The
  client (`OpenRouterDecisionClient`) sits next to `get_chat_model()` in
  `backend/config/llm_clients.py`, behind a one-method `DecisionClient`
  interface. TypeSafe's own API or a self-hosted `/v1/systemone` endpoint
  would be another implementation chosen in `get_decision_client()`;
  nothing else would change. `typesafe_sdk` isn't a dependency.
- **A second kind of option in the intent-classification slot, not a new
  config path.** `model_assignments` gains `model_kind` (`chat` or
  `decision`) and the two thresholds below (migration `b5f05639dedf`). Only
  slots whose definition sets `accepts_decision_models` offer one; today
  that's intent classification alone. An unset slot, or a chat model in it,
  behaves exactly as before.
- **Pinned dated ids.** The picker lists decision models by
  `canonical_slug` and never offers a `~` alias. Saving (or a harness run)
  by another listed id stores the dated one.
- **One Choice per decision, not multi-label Noul.** Choice probabilities
  are well calibrated (pooled ECE 0.028 in the Jev paper), while Noul
  over-predicts yes and needs per-label thresholds tuned on ~1,000
  examples. We have 35 labelled rows.
  - *Family*: one option per family display name, plus `unclear` (points
    at a family but not exactly one) and `no_family` (names none, which is
    what keeps a follow-up on the previous family, #22). A family is
    picked when it's the answer with probability at or above the **family
    confidence threshold**; otherwise the rep is asked, offered the
    families above 5% (most likely first, at most four).
  - *Question type*: one option per question-type slug, criteria from
    their descriptions. When the runner-up is within the **question-type
    margin** of the answer, both count as detected and #19's clarification
    fires. `unclassified` is never paired.
  - *Question-type reply*: one Choice over the same options, state
    `{question, reply}`.
  - The answers are returned in `_IntentClassification`'s shape, so
    `detect_intent`'s pause conditions, and the intent eval through them,
    are unchanged.
- **Thresholds are stored with the slot**, with starting defaults of
  **0.6** (family confidence) and **0.15** (question-type margin) in
  `model_assignments/slots.py`. They aren't tuned. Tune them on the intent
  eval with `--family-confidence-threshold` and `--question-type-margin`.
- **Its own probe on save.** Detect_intent's real Choice request must come
  back with a family answer that is one of its own keys, with
  probabilities, and so must the question type and the question-type reply
  (`probe.py`'s `DECISION_SLOT_CHECKS`).
- **Any decision failure falls back to the chat path.** An HTTP error,
  timeout, 402/429, or an answer that isn't one of the question's options
  raises `DecisionModelError`, logs `slot_output_fallback` (with
  `model_kind=decision`), and runs the slot's default call on
  `CHAT_MODEL`. There's no OpenAI-direct decision API, so this is also the
  outage fallback.

## How to compare the two paths

From `backend/`, with `OPENAI_API_KEY`, `OPENROUTER_API_KEY` and a seeded
catalog mirror:

```
python -m evals.harness intent --output intent-chat.json
python -m evals.harness intent --model intent_classification=<dated decision model id> \
    --output intent-decision.json
python -m evals.harness intent --model intent_classification=<dated decision model id> \
    --family-confidence-threshold 0.5 --question-type-margin 0.1
```

The report adds the decision model's tokens and cost per turn (from the
response's `usage`). Ship only if family accuracy and family-clarification
precision don't regress against the chat path.

## Consequences

- Rep queries go to a new model vendor through OpenRouter. They're product
  questions, not patient data, but check TypeSafe's (and Microsoft's)
  retention terms before choosing one in production.
- The Decisions API is alpha. A breaking change shows up as
  `slot_output_fallback` lines, and turns keep working on `CHAT_MODEL`.
- Nothing here was run against the live API: the client, probe, routes and
  harness are tested against mocked HTTP and fakes. The first real call
  happens when an admin tests a decision model in Settings, or runs the
  intent eval with one.
- The family candidates' 5% floor (`CANDIDATE_PROBABILITY_FLOOR`) isn't a
  tuned threshold. It only trims options a rep would never mean.

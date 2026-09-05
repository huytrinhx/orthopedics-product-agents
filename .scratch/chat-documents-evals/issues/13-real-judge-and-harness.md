# 13: Evals — real judge and harness

**What to build:** `backend/agents/judge.py`'s `judge_answer` and `backend/evals/harness.py`'s `run_eval` actually work, so a golden dataset can be scored against a real workflow from the command line — the foundation the admin dashboard (ticket 15) will sit on top of.

**Blocked by:** 01 (Add Postgres migration tooling and a users table), 08 (Chat baseline workflow, end-to-end)

**Status:** shelved (2026-09-05, grilling session), not deleted — see below.

- [x] `judge_answer(query, retrieved, answer)` calls an LLM to score faithfulness/relevance/style/citation and returns real `EvalScores`, not `NotImplementedError` — done organically over the session as a shared module (backend/agents/judge.py), used by every workflow's own self_eval node and the human feedback comparison, long before this ticket was ever picked up directly. This ticket's premise ("judge_answer doesn't work yet") was stale by the time it came up for grilling.
- [ ] `run_eval(workflow_name, dataset_path)` -- still `NotImplementedError` (`backend/evals/harness.py`).
- [ ] Everything else below -- not started.

**Why shelved:** grilled 2026-09-05 to scope the actual harness work, and the premise broke immediately: every workflow now has a real clarification-pause path (detect_intent's system disambiguation, resolve_synonyms' term-ambiguity check, self_eval's low-confidence gate) that can suspend a turn via `interrupt()` instead of ever producing an `answer`. A live rerun of `mis.jsonl` that same day (the "MIS Eval Reflection" artifact) hit this directly: 6 of 11 golden questions paused rather than answered. A batch harness scoring a static golden dataset needs a real, deliberate policy for "what does it mean to score a pause" (skip it? score the pause itself as right/wrong? fabricate a resume reply?) that didn't exist yet and isn't cheap to get right blind.

Rather than build that policy speculatively, the session pivoted to a smaller, more immediately useful capability instead: ticket 15, rescoped the same day from "eval-dataset dashboard" into a feedback-rerun tool -- re-run one specific real (already-completed) flagged conversation against the current code, in its real context, sharing an admin's own live clarification answer if one comes up again. That sidesteps the "what does a golden-set pause even mean" problem entirely, since a rerun's clarification pauses are answered by a real person in the moment, not scored blind.

**Revisit condition:** once there's a clear answer for how a batch/CI harness should treat a clarification pause (and, separately, once `evals/golden_datasets/build_dataset.py`'s `expected_question_type` is reconciled with `agents/question_types.py`'s canonical slugs -- a related, also-still-open mismatch, see that file's own docstring), this is worth picking back up. Until then, ticket 15's rerun tool covers the actual day-to-day need ("did this fix work") this ticket was meant to serve.

**Deferred to a future ticket (unchanged from before):** the `evals` CI job runs against a fresh, empty Postgres/Neo4j (no ingested documents) with no `OPENAI_API_KEY` or DB services configured for that job at all (confirmed by reading `.github/workflows/ci.yml` directly during the grilling session) -- so `deterministic` couldn't even connect to a database today, let alone produce a meaningful score. Whoever eventually resumes this ticket needs to fix that job's config too, not just `run_eval` itself.

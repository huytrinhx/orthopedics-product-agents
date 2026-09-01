# 13: Evals — real judge and harness

**What to build:** `backend/agents/judge.py`'s `judge_answer` and `backend/evals/harness.py`'s `run_eval` actually work, so a golden dataset can be scored against a real workflow from the command line — the foundation the admin dashboard (ticket 15) will sit on top of.

**Blocked by:** 01 (Add Postgres migration tooling and a users table), 08 (Chat baseline workflow, end-to-end)

**Status:** ready-for-agent

- [ ] `judge_answer(query, retrieved, answer)` calls an LLM to score faithfulness/relevance/style/citation and returns real `EvalScores`, not `NotImplementedError`
- [ ] `run_eval(workflow_name, dataset_path)` loads a JSONL golden dataset, runs each example through the named registered workflow, scores it with `judge_answer`, and returns per-example + aggregate results
- [ ] Running this via CLI against `backend/evals/golden_datasets/mis.jsonl` (or `reflex.jsonl`) with `deterministic` produces real, sensible scores
- [ ] The existing CI job (`.github/workflows/ci.yml`'s `evals` job, `--all-workflows`) still runs and doesn't fail purely because two of the three workflows remain unimplemented — skip or clearly report those rather than erroring the whole run

# 15: Evals results persistence and dashboard

**What to build:** Eval runs stop disappearing into CI logs. Results persist, and an admin can see run history — per system, per workflow — on a dashboard page.

**Blocked by:** 01 (Add Postgres migration tooling and a users table), 02 (Email/password auth with admin flag), 13 (Evals — real judge and harness)

**Status:** ready-for-agent

- [ ] A migration adds tables for eval runs and per-example results (run id, workflow, dataset/system, timestamp, aggregate scores; per-example query/scores/citations)
- [ ] `run_eval` (ticket 13) writes its results into these tables instead of only returning them in-memory
- [ ] The existing CI-triggered run (`.github/workflows/ci.yml`) also writes to this same table, so CI and any future manual runs share one history
- [ ] An admin-only `/evals` dashboard page lists past runs and shows per-system/per-workflow aggregate scores, with the ability to drill into a run's per-example results

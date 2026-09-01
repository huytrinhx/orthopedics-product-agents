# 16: Evals "run now" trigger

**What to build:** An admin can kick off a golden-dataset eval run on demand from the dashboard, instead of waiting for CI.

**Blocked by:** 15 (Evals results persistence and dashboard)

**Status:** ready-for-agent

- [ ] The dashboard has a "run now" control letting an admin pick a dataset + workflow and start a run
- [ ] The run executes as a background task (mirroring the ingestion background-task pattern from ticket 04) and its status is visible while in progress
- [ ] On completion it appears in the same run history as CI-triggered runs, indistinguishable in structure (just distinguishable by trigger source)
- [ ] Two runs can't corrupt each other's results if triggered close together

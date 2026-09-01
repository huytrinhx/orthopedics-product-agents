# 17: Evals feedback review and promote-to-golden-dataset

**What to build:** An admin can review flagged/low-scored chat feedback and promote it into the golden-dataset pipeline's source of truth — replacing `feedback-notes.csv` with a Postgres table, per the decision made in the grilling session that produced this ticket set (this supersedes the current line in `agents.md` describing the CSV as source of truth).

**Blocked by:** 01 (Add Postgres migration tooling and a users table), 02 (Email/password auth with admin flag), 11 (Chat inline per-message 4-axis feedback)

**Status:** ready-for-agent

- [ ] A migration adds a `feedback_notes` table shaped like `feedback-notes.csv` (prompt/provided answer/preferred answer/content+formatting feedback/system), sourced from promoted chat feedback
- [ ] An admin-only review UI lists flagged and/or low-scored chat feedback (from ticket 11/12) and lets the admin "promote" an entry into `feedback_notes`
- [ ] `backend/evals/golden_datasets/build_dataset.py` is updated to read from the `feedback_notes` table instead of the CSV, and still requires an explicit manual run to regenerate the per-system JSONL (no auto-regeneration on promote)
- [ ] `agents.md` and the relevant ADR are updated to reflect Postgres as the new source of truth, with a note on why (this ticket) and what changed

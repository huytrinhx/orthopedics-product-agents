# 11: Chat inline per-message 4-axis feedback

**What to build:** A user can score any assistant message on the same four axes the judge uses (faithfulness, relevance, style, citation), directly under that message. This wires up `backend/api/routes/feedback.py`'s `submit_feedback`, which currently raises `NotImplementedError`, to real persistence.

**Blocked by:** 01 (Add Postgres migration tooling and a users table), 08 (Chat baseline workflow, end-to-end)

**Status:** ready-for-agent

- [ ] A migration adds a table for submitted feedback (thread_id, message_id, flagged, the 4 scores, optional comment, submitted_by, created_at)
- [ ] Each assistant message has a compact 4-axis scoring control (matching `EvalScores`' shape) plus an optional comment and a "flagged" toggle
- [ ] Submitting writes a real row via `submit_feedback`, keyed to that specific `message_id`
- [ ] A user can see that their feedback on a given message was recorded (e.g. the control reflects the submitted state)

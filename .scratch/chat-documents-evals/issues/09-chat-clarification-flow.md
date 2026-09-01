# 09: Chat clarification flow

**What to build:** When the agent can't confidently tell which product system a query is about, it asks instead of guessing — using LangGraph's `interrupt()`/resume pattern — and the UI surfaces that as a clarifying question.

**Blocked by:** 08 (Chat baseline workflow, end-to-end)

**Status:** ready-for-agent

- [ ] The workflow calls `interrupt()` when intent detection is ambiguous, matching the `expects_clarification` cases in `intent_detection.jsonl`
- [ ] `POST /chat/{workflow_name}/resume` resumes the suspended graph with the user's answer
- [ ] The chat UI shows the clarifying question, with any suggested options as clickable buttons plus a free-text fallback
- [ ] Answering (by button or free text) resumes the same conversation turn rather than starting a new one
- [ ] An ambiguous query from the golden dataset (e.g. one flagged `expects_clarification: true`) is verified to actually trigger this path end-to-end

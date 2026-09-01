# 12: Chat free-text "Give feedback" entry

**What to build:** A standalone feedback entry point, separate from the per-message scoring, prompting "What can I do better?" for open-ended conversation-level feedback.

**Blocked by:** 11 (Chat inline per-message 4-axis feedback)

**Status:** ready-for-agent

- [ ] A "Give feedback" button is available during/after a conversation
- [ ] Clicking it opens a free-text-only prompt: "What can I do better?"
- [ ] Submitting reuses the same feedback persistence as ticket 11 (comment populated, scores omitted), attributed to the conversation's latest message
- [ ] The user sees confirmation that their feedback was recorded

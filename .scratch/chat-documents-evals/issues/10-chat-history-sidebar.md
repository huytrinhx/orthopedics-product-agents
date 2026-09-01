# 10: Chat persisted history sidebar

**What to build:** A logged-in user can see, select, and resume their past conversations, rather than losing them on refresh.

**Blocked by:** 08 (Chat baseline workflow, end-to-end)

**Status:** ready-for-agent

- [ ] Past threads for the logged-in user are listed in a sidebar (most recent first), scoped to their `user_id`
- [ ] Selecting a past thread loads its transcript (from the Postgres checkpointer state) and lets the user continue it
- [ ] Starting a new conversation is a clearly separate action from resuming an old one
- [ ] One user never sees another user's threads

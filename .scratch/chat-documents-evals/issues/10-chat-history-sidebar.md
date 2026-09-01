# 10: Chat persisted history sidebar

**What to build:** A logged-in user can see, select, and resume their past conversations, rather than losing them on refresh.

**Blocked by:** 08 (Chat baseline workflow, end-to-end)

**Status:** done

- [x] Past threads for the logged-in user are listed in a sidebar (most recent first), scoped to their `user_id`
- [x] Selecting a past thread loads its transcript (from the Postgres checkpointer state) and lets the user continue it
- [x] Starting a new conversation is a clearly separate action from resuming an old one
- [x] One user never sees another user's threads

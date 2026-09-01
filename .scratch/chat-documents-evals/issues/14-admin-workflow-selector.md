# 14: Admin workflow-selector config

**What to build:** An admin page where the single global default chat workflow is chosen — the only place workflow choice is ever exposed; regular chat users never see it.

**Blocked by:** 02 (Email/password auth with admin flag), 08 (Chat baseline workflow, end-to-end)

**Status:** done

- [x] An admin-only settings page lists the registered workflows (`backend/agents/registry.py`'s `list_workflows()`) and lets the admin pick one as the default
- [x] `deterministic` is the only one actually selectable/functional today — `react_agent` and `supervisor` remain unimplemented stubs and are out of scope for this ticket (either hidden or clearly marked unavailable in the picker)
- [x] `POST /chat/{workflow_name}/stream` uses the admin-configured default when the chat UI doesn't explicitly pick one, rather than a hardcoded value
- [x] Changing the setting takes effect for new conversations without a redeploy

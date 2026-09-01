"""Exercises backend/api/routes/admin.py (ticket 14: admin workflow-selector
config) against the real app and real Postgres -- same approach as
test_tags.py. No LLM/OPENAI_API_KEY needed: nothing here runs a workflow,
just reads/writes the app_settings row and validates against the registry.
"""
import uuid

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _admin_token(monkeypatch) -> str:
    email = _unique_email()
    monkeypatch.setenv("ADMIN_EMAILS", email)
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return res.json()["access_token"]


def _user_token() -> str:
    email = _unique_email()
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return res.json()["access_token"]


def test_get_settings_requires_auth():
    res = client.get("/admin/settings")
    assert res.status_code == 401


def test_get_settings_requires_admin():
    token = _user_token()
    res = client.get("/admin/settings", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403


def test_get_settings_lists_all_registered_workflows_with_functional_flags(monkeypatch):
    token = _admin_token(monkeypatch)
    res = client.get("/admin/settings", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    body = res.json()
    assert body["default_workflow"] == "deterministic"
    by_name = {wf["name"]: wf["functional"] for wf in body["workflows"]}
    assert by_name == {"deterministic": True, "react_agent": False, "supervisor": False}


def test_put_settings_requires_admin():
    token = _user_token()
    res = client.put(
        "/admin/settings",
        headers={"Authorization": f"Bearer {token}"},
        json={"default_workflow": "deterministic"},
    )
    assert res.status_code == 403


def test_put_settings_rejects_unknown_workflow(monkeypatch):
    token = _admin_token(monkeypatch)
    res = client.put(
        "/admin/settings",
        headers={"Authorization": f"Bearer {token}"},
        json={"default_workflow": "not-a-real-workflow"},
    )
    assert res.status_code == 404


def test_put_settings_rejects_a_non_functional_workflow(monkeypatch):
    """react_agent/supervisor are registered (so the picker can show them,
    disabled) but their build_graph is still a stub -- picking either as the
    live default would break every new conversation, so the route rejects
    it before it's ever persisted.
    """
    token = _admin_token(monkeypatch)
    res = client.put(
        "/admin/settings",
        headers={"Authorization": f"Bearer {token}"},
        json={"default_workflow": "react_agent"},
    )
    assert res.status_code == 400


def test_put_settings_updates_the_default_and_get_reflects_it(monkeypatch):
    token = _admin_token(monkeypatch)
    headers = {"Authorization": f"Bearer {token}"}
    # Only "deterministic" is functional today, so this is a no-op change in
    # value but still exercises the real UPDATE + re-read round trip.
    put_res = client.put("/admin/settings", headers=headers, json={"default_workflow": "deterministic"})
    assert put_res.status_code == 200
    assert put_res.json()["default_workflow"] == "deterministic"

    get_res = client.get("/admin/settings", headers=headers)
    assert get_res.json()["default_workflow"] == "deterministic"

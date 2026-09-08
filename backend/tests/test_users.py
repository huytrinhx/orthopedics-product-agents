"""Exercises the admin-only Users tab endpoints (GET /users/, PATCH
/users/{id}/active) against a real Postgres -- same approach as
test_documents.py/test_auth.py.
"""
import uuid

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _admin(monkeypatch) -> dict:
    email = _unique_email()
    monkeypatch.setenv("ADMIN_EMAILS", email)
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return res.json()


def _user(monkeypatch) -> dict:
    email = _unique_email()
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return res.json()


def test_list_users_requires_admin(monkeypatch):
    token = _user(monkeypatch)["access_token"]
    res = client.get("/users/", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403


def test_list_users_includes_admins_and_pending_users(monkeypatch):
    admin = _admin(monkeypatch)
    pending = _user(monkeypatch)
    admin_headers = {"Authorization": f"Bearer {admin['access_token']}"}

    res = client.get("/users/", headers=admin_headers)
    assert res.status_code == 200
    by_id = {u["id"]: u for u in res.json()}

    assert by_id[admin["user"]["id"]]["is_admin"] is True
    assert by_id[admin["user"]["id"]]["is_active"] is True
    assert by_id[pending["user"]["id"]]["is_admin"] is False
    assert by_id[pending["user"]["id"]]["is_active"] is False


def test_set_active_requires_admin(monkeypatch):
    actor = _user(monkeypatch)
    target = _user(monkeypatch)
    res = client.patch(
        f"/users/{target['user']['id']}/active",
        headers={"Authorization": f"Bearer {actor['access_token']}"},
        json={"is_active": True},
    )
    assert res.status_code == 403


def test_admin_can_enable_and_disable_a_user(monkeypatch):
    admin = _admin(monkeypatch)
    target = _user(monkeypatch)
    admin_headers = {"Authorization": f"Bearer {admin['access_token']}"}

    enable = client.patch(
        f"/users/{target['user']['id']}/active", headers=admin_headers, json={"is_active": True}
    )
    assert enable.status_code == 200
    assert enable.json()["is_active"] is True

    disable = client.patch(
        f"/users/{target['user']['id']}/active", headers=admin_headers, json={"is_active": False}
    )
    assert disable.status_code == 200
    assert disable.json()["is_active"] is False


def test_set_active_on_unknown_user_404s(monkeypatch):
    admin = _admin(monkeypatch)
    res = client.patch(
        f"/users/{uuid.uuid4()}/active",
        headers={"Authorization": f"Bearer {admin['access_token']}"},
        json={"is_active": True},
    )
    assert res.status_code == 404


def test_admin_rows_cannot_be_disabled(monkeypatch):
    """Admin rows are read-only in the Users tab -- this is the backend's
    own defense in depth for that, independent of the frontend hiding the
    button (require_chat_access never even consults is_active for an admin,
    so honoring this would silently do nothing while implying it worked).
    """
    admin = _admin(monkeypatch)
    other_admin_email = _unique_email()
    monkeypatch.setenv("ADMIN_EMAILS", f"{admin['user']['email']}, {other_admin_email}")
    other_admin = client.post(
        "/auth/signup", json={"email": other_admin_email, "password": "correct horse battery"}
    ).json()

    res = client.patch(
        f"/users/{other_admin['user']['id']}/active",
        headers={"Authorization": f"Bearer {admin['access_token']}"},
        json={"is_active": False},
    )
    assert res.status_code == 400

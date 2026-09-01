"""Exercises the signup/login/me flow against a real Postgres (DATABASE_URL,
see README's local-dev setup / CI's postgres service) — no mocking of the
users table, since the point is to prove the real SQL round-trips.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import app
from auth import oauth
from auth.security import create_oauth_state

client = TestClient(app)


@pytest.fixture
def mock_google(monkeypatch):
    """Stubs the two outbound calls in auth/oauth.py so tests never hit the
    real Google endpoints -- everything else (state verification, user
    lookup/creation, allowlist, session issuance) runs for real.
    """

    def _install(email: str, email_verified: bool = True):
        async def fake_exchange(code: str) -> str:
            return f"fake-access-token-for-{code}"

        async def fake_fetch(access_token: str) -> str:
            if not email_verified:
                raise ValueError("Google account email is not verified")
            return email

        monkeypatch.setattr(oauth, "exchange_code_for_access_token", fake_exchange)
        monkeypatch.setattr(oauth, "fetch_verified_email", fake_fetch)

    return _install


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def test_signup_then_me():
    email = _unique_email()
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    assert res.status_code == 200
    body = res.json()
    assert body["user"]["email"] == email
    assert body["user"]["is_admin"] is False

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_signup_duplicate_email_conflicts():
    email = _unique_email()
    client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    res = client.post("/auth/signup", json={"email": email, "password": "another password"})
    assert res.status_code == 409


def test_login_wrong_password_rejected():
    email = _unique_email()
    client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    res = client.post("/auth/login", json={"email": email, "password": "wrong password"})
    assert res.status_code == 401


def test_login_correct_password_succeeds():
    email = _unique_email()
    client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    res = client.post("/auth/login", json={"email": email, "password": "correct horse battery"})
    assert res.status_code == 200
    assert res.json()["user"]["email"] == email


def test_me_requires_a_token():
    assert client.get("/auth/me").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"}).status_code == 401


def test_admin_emails_allowlist_grants_is_admin(monkeypatch):
    email = _unique_email()
    monkeypatch.setenv("ADMIN_EMAILS", f"someone-else@example.com, {email}")
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    assert res.json()["user"]["is_admin"] is True


def test_google_login_redirects_to_google_with_signed_state():
    res = client.get("/auth/google/login", follow_redirects=False)
    assert res.status_code in (302, 307)
    assert res.headers["location"].startswith("https://accounts.google.com/o/oauth2/v2/auth")


def test_google_callback_rejects_bad_state():
    res = client.get(
        "/auth/google/callback",
        params={"code": "some-code", "state": "not-a-real-state"},
        follow_redirects=False,
    )
    assert res.status_code == 400


def test_google_callback_creates_oauth_only_user_and_hands_back_session(mock_google):
    email = _unique_email()
    mock_google(email)

    res = client.get(
        "/auth/google/callback",
        params={"code": "some-code", "state": create_oauth_state()},
        follow_redirects=False,
    )
    assert res.status_code in (302, 307)
    location = res.headers["location"]
    assert location.startswith("/?auth_token=") or "?auth_token=" in location

    token = location.split("auth_token=", 1)[1]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_google_callback_links_to_existing_password_account(mock_google):
    email = _unique_email()
    client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    mock_google(email)

    res = client.get(
        "/auth/google/callback",
        params={"code": "some-code", "state": create_oauth_state()},
        follow_redirects=False,
    )
    token = res.headers["location"].split("auth_token=", 1)[1]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == email

    # still able to log in with the original password -- linking didn't
    # clobber the existing account
    login_res = client.post("/auth/login", json={"email": email, "password": "correct horse battery"})
    assert login_res.status_code == 200


def test_google_callback_applies_admin_allowlist(mock_google, monkeypatch):
    email = _unique_email()
    monkeypatch.setenv("ADMIN_EMAILS", email)
    mock_google(email)

    res = client.get(
        "/auth/google/callback",
        params={"code": "some-code", "state": create_oauth_state()},
        follow_redirects=False,
    )
    token = res.headers["location"].split("auth_token=", 1)[1]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["is_admin"] is True


def test_google_callback_rejects_unverified_email(mock_google):
    email = _unique_email()
    mock_google(email, email_verified=False)

    with pytest.raises(ValueError):
        client.get(
            "/auth/google/callback",
            params={"code": "some-code", "state": create_oauth_state()},
        )

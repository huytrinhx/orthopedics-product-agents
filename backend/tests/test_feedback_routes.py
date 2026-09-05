"""Exercises backend/api/routes/feedback.py against the real app and a real
Postgres DB -- no mocking, same approach as test_documents.py/test_chat_routes.py.
Ownership is checked from the thread_id prefix alone (chat_threads.repository.
owns_thread), before any DB write, so most of this doesn't need a real chat
turn or OPENAI_API_KEY -- only a message_id (any string the frontend would
have gotten from the backend in real usage) and a thread_id shaped like a
real one. The one thing that does need a real row: `feedback.thread_id` is
FK'd to `chat_threads.thread_id` (backend/migrations/versions/
5d95b6897886_create_feedback_table.py), so tests that actually write a row
create a chat_threads row directly via chat_threads.repository.create_thread
first, mirroring what stream_chat does for a real turn.
"""
import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import app
from chat_threads.repository import create_thread

needs_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="requires a real OPENAI_API_KEY"
)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _signup(client: TestClient) -> dict:
    email = _unique_email()
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return res.json()


def _sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    event_name = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event_name = line[len("event: ") :]
        elif line.startswith("data: "):
            events.append((event_name, json.loads(line[len("data: ") :])))
    return events


def test_submit_feedback_requires_auth():
    with TestClient(app) as client:
        res = client.post(
            "/feedback/",
            json={
                "thread_id": f"anyone:{uuid.uuid4().hex}",
                "message_id": str(uuid.uuid4()),
                "flagged": False,
                "scores": {},
            },
        )
    assert res.status_code == 401


def test_submit_feedback_someone_elses_thread_is_rejected():
    with TestClient(app) as client:
        user_a = _signup(client)
        user_b = _signup(client)
        headers_b = {"Authorization": f"Bearer {user_b['access_token']}"}
        # Correctly shaped (owner-prefixed) but belongs to user_a -- rejected
        # from the thread_id prefix alone, before ever touching the DB, so no
        # chat_threads row is needed for this case.
        thread_id = f"{user_a['user']['id']}:{uuid.uuid4().hex}"
        res = client.post(
            "/feedback/",
            headers=headers_b,
            json={
                "thread_id": thread_id,
                "message_id": str(uuid.uuid4()),
                "flagged": False,
                "scores": {"faithfulness": 1.0},
            },
        )
    assert res.status_code == 403


async def test_submit_feedback_writes_a_real_row_keyed_to_message_id():
    with TestClient(app) as client:
        user = _signup(client)
        user_id = uuid.UUID(user["user"]["id"])
        headers = {"Authorization": f"Bearer {user['access_token']}"}
        thread_id = f"{user_id}:{uuid.uuid4().hex}"
        message_id = str(uuid.uuid4())
        await create_thread(thread_id, user_id, "test thread")

        res = client.post(
            "/feedback/",
            headers=headers,
            json={
                "thread_id": thread_id,
                "message_id": message_id,
                "flagged": True,
                "scores": {"faithfulness": 0.9, "relevance": 0.8, "style": 0.7, "citation": 1.0},
                "comment": "cites the wrong torque spec",
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["message_id"] == message_id
    assert body["thread_id"] == thread_id
    assert body["flagged"] is True
    assert body["scores"] == {
        "faithfulness": 0.9,
        "relevance": 0.8,
        "style": 0.7,
        "citation": 1.0,
    }
    assert body["comment"] == "cites the wrong torque spec"
    assert body["submitted_by"] == str(user_id)


async def test_submitting_again_on_the_same_message_overwrites_not_duplicates():
    """Ticket 11's design: resubmitting on the same message_id is an upsert
    (ON CONFLICT (message_id) DO UPDATE, feedback/repository.py), not a
    second row -- a user correcting a misclick just overwrites."""
    with TestClient(app) as client:
        user = _signup(client)
        user_id = uuid.UUID(user["user"]["id"])
        headers = {"Authorization": f"Bearer {user['access_token']}"}
        thread_id = f"{user_id}:{uuid.uuid4().hex}"
        message_id = str(uuid.uuid4())
        await create_thread(thread_id, user_id, "test thread")

        first = client.post(
            "/feedback/",
            headers=headers,
            json={
                "thread_id": thread_id,
                "message_id": message_id,
                "flagged": False,
                "scores": {"faithfulness": 0.2},
            },
        )
        second = client.post(
            "/feedback/",
            headers=headers,
            json={
                "thread_id": thread_id,
                "message_id": message_id,
                "flagged": True,
                "scores": {"faithfulness": 0.9},
            },
        )
    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["flagged"] is True
    assert second.json()["scores"] == {"faithfulness": 0.9}
    # Same row, not a new one -- created_at is stable across the upsert.
    assert first.json()["created_at"] == second.json()["created_at"]


async def test_submit_feedback_accepts_a_flag_or_comment_with_no_scores_at_all():
    """Ticket 12 (free-text-only "Give feedback") reuses this same endpoint
    with scores omitted entirely -- must not be forced to send all four."""
    with TestClient(app) as client:
        user = _signup(client)
        user_id = uuid.UUID(user["user"]["id"])
        headers = {"Authorization": f"Bearer {user['access_token']}"}
        thread_id = f"{user_id}:{uuid.uuid4().hex}"
        message_id = str(uuid.uuid4())
        await create_thread(thread_id, user_id, "test thread")

        res = client.post(
            "/feedback/",
            headers=headers,
            json={
                "thread_id": thread_id,
                "message_id": message_id,
                "flagged": False,
                "comment": "What can I do better? More citations please.",
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["scores"] == {}
    assert body["comment"] == "What can I do better? More citations please."


def test_flagged_list_requires_admin():
    with TestClient(app) as client:
        token = _signup(client)["access_token"]
        res = client.get("/feedback/flagged", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403


def test_resolved_toggle_requires_admin():
    with TestClient(app) as client:
        token = _signup(client)["access_token"]
        res = client.patch(
            f"/feedback/{uuid.uuid4()}/resolved",
            headers={"Authorization": f"Bearer {token}"},
            json={"resolved": True},
        )
    assert res.status_code == 403


async def test_resolved_toggle_updates_the_row_and_survives_a_feedback_resubmit(monkeypatch):
    """resolved is a separate admin action, not part of what a user submits
    -- resubmitting feedback on the same message (ticket 11's upsert) must
    not silently reset it back to false."""
    with TestClient(app) as client:
        user = _signup(client)
        user_id = uuid.UUID(user["user"]["id"])
        user_headers = {"Authorization": f"Bearer {user['access_token']}"}
        thread_id = f"{user_id}:{uuid.uuid4().hex}"
        message_id = str(uuid.uuid4())
        await create_thread(thread_id, user_id, "test thread")

        admin_email = _unique_email()
        monkeypatch.setenv("ADMIN_EMAILS", admin_email)
        admin_token = client.post(
            "/auth/signup", json={"email": admin_email, "password": "correct horse battery"}
        ).json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        client.post(
            "/feedback/",
            headers=user_headers,
            json={"thread_id": thread_id, "message_id": message_id, "flagged": True, "scores": {}},
        )

        toggled = client.patch(
            f"/feedback/{message_id}/resolved", headers=admin_headers, json={"resolved": True}
        )
        assert toggled.status_code == 200
        assert toggled.json()["resolved"] is True

        resubmit = client.post(
            "/feedback/",
            headers=user_headers,
            json={"thread_id": thread_id, "message_id": message_id, "flagged": True, "scores": {}},
        )
    assert resubmit.json()["resolved"] is True


@needs_openai_key
def test_flagged_feedback_lists_the_actual_question_and_answer_text(monkeypatch):
    with TestClient(app) as client:
        admin_email = _unique_email()
        monkeypatch.setenv("ADMIN_EMAILS", admin_email)
        admin_token = client.post(
            "/auth/signup", json={"email": admin_email, "password": "correct horse battery"}
        ).json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        rep_token = _signup(client)["access_token"]
        rep_headers = {"Authorization": f"Bearer {rep_token}"}

        question = "What torque should I use on a REFLEX HYBRID locking screw?"
        stream = client.post(
            "/chat/deterministic/stream", headers=rep_headers, json={"message": question}
        )
        events = dict(_sse_events(stream.text))
        thread_id = events["thread"]["thread_id"]
        message_id = events["done"]["message_id"]
        answer = events["done"]["answer"]

        client.post(
            "/feedback/",
            headers=rep_headers,
            json={"thread_id": thread_id, "message_id": message_id, "flagged": True, "scores": {}},
        )

        flagged = client.get("/feedback/flagged", headers=admin_headers).json()

    row = next(r for r in flagged if r["message_id"] == message_id)
    assert row["question"] == question
    assert row["answer"] == answer
    assert row["flagged"] is True
    assert row["resolved"] is False

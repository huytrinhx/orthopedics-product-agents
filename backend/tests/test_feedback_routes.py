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
import uuid

from fastapi.testclient import TestClient

from api.main import app
from chat_threads.repository import create_thread


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _signup(client: TestClient) -> dict:
    email = _unique_email()
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return res.json()


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

"""Exercises backend/api/routes/chat.py's streaming endpoint against the
real app -- no mocking, same approach as test_documents.py/test_tags.py.
Route-level concerns (auth, unknown workflow, thread ownership) don't need
a real LLM; the full pipeline does, so that one test is gated behind
OPENAI_API_KEY like test_entity_extraction.py.
"""
import json
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import app

needs_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="requires a real OPENAI_API_KEY"
)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _user_token(client: TestClient) -> str:
    email = _unique_email()
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return res.json()["access_token"]


def _sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    event_name = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event_name = line[len("event: ") :]
        elif line.startswith("data: "):
            events.append((event_name, json.loads(line[len("data: ") :])))
    return events


def test_stream_requires_auth():
    with TestClient(app) as client:
        res = client.post("/chat/deterministic/stream", json={"message": "hi"})
    assert res.status_code == 401


def test_stream_unknown_workflow_404s():
    with TestClient(app) as client:
        token = _user_token(client)
        res = client.post(
            "/chat/not-a-real-workflow/stream",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": "hi"},
        )
    assert res.status_code == 404


@needs_openai_key
def test_someone_elses_thread_id_is_rejected():
    with TestClient(app) as client:
        token_a = _user_token(client)
        token_b = _user_token(client)

        first = client.post(
            "/chat/deterministic/stream",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"message": "hello"},
        )
        thread_id = _sse_events(first.text)[0][1]["thread_id"]

        second = client.post(
            "/chat/deterministic/stream",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"message": "hello", "thread_id": thread_id},
        )
    assert second.status_code == 403


@needs_openai_key
def test_full_pipeline_answers_with_citations_from_a_real_indexed_document(monkeypatch, tmp_path):
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    with TestClient(app) as client:
        admin_email = _unique_email()
        monkeypatch.setenv("ADMIN_EMAILS", admin_email)
        admin_token = client.post(
            "/auth/signup", json={"email": admin_email, "password": "correct horse battery"}
        ).json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        content = (
            b"REFLEX HYBRID Torque Spec\n\n"
            b"The REFLEX HYBRID driver requires 1.2 Nm of torque when seating the "
            b"4.0mm locking screw. Do not exceed 1.5 Nm or the screw head may strip."
        )
        upload = client.post(
            "/documents/upload",
            headers=admin_headers,
            files={"file": ("torque-spec.txt", content, "text/plain")},
        )
        doc_id = upload.json()["id"]
        client.post(f"/documents/{doc_id}/index", headers=admin_headers)
        for _ in range(40):
            if client.get(f"/documents/{doc_id}", headers=admin_headers).json()["status"] == "done":
                break
            time.sleep(0.25)

        user_token = _user_token(client)
        res = client.post(
            "/chat/deterministic/stream",
            headers={"Authorization": f"Bearer {user_token}"},
            json={"message": "How much torque should I use on the REFLEX HYBRID locking screw?"},
        )
        done = dict(_sse_events(res.text))["done"]

        client.delete(f"/documents/{doc_id}", headers=admin_headers)

    assert "1.2" in done["answer"]
    assert done["citations"]
    assert all(citation.startswith(doc_id) for citation in done["citations"])
    assert done["eval_scores"]["faithfulness"] > 0.5

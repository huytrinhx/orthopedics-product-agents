"""Exercises backend/api/routes/chat.py's streaming and threads endpoints
against the real app -- no mocking, same approach as
test_documents.py/test_tags.py. Route-level concerns (auth, unknown
workflow, thread ownership) don't need a real LLM; anything that has to
complete an actual chat turn does, so those are gated behind OPENAI_API_KEY
like test_entity_extraction.py.
"""
import json
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

import agents.workflows.deterministic as det
from api.main import app

needs_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="requires a real OPENAI_API_KEY"
)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _signup(client: TestClient) -> dict:
    email = _unique_email()
    res = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return res.json()


def _user_token(client: TestClient) -> str:
    return _signup(client)["access_token"]


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


def test_stream_default_requires_auth():
    with TestClient(app) as client:
        res = client.post("/chat/stream", json={"message": "hi"})
    assert res.status_code == 401


def test_stream_default_resolves_the_admin_configured_workflow(monkeypatch):
    """POST /chat/stream (ticket 14) has no workflow_name of its own -- it
    reads settings/repository.py's get_settings() and delegates to the same
    validation _stream_chat does for the explicit route. Proven here without
    a real LLM turn (or a second functional workflow to switch to) by
    pointing the configured default at a bogus name and checking it 404s
    exactly like /chat/not-a-real-workflow/stream does above -- that 404
    could only come from the value this test set flowing all the way
    through.
    """
    import api.routes.chat as chat_routes
    from settings.repository import AppSettingsRecord

    async def _fake_get_settings():
        return AppSettingsRecord(default_workflow="not-a-real-workflow", updated_at=None, updated_by=None)

    monkeypatch.setattr(chat_routes, "get_settings", _fake_get_settings)
    with TestClient(app) as client:
        token = _user_token(client)
        res = client.post(
            "/chat/stream", headers={"Authorization": f"Bearer {token}"}, json={"message": "hi"}
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
    assert all(citation["document_id"] == doc_id for citation in done["citations"])
    assert all(citation["filename"] == "torque-spec.txt" for citation in done["citations"])
    assert done["eval_scores"]["faithfulness"] > 0.5
    # Only asserted when Langfuse is actually configured (see
    # observability/langfuse_setup.py) -- the SDK no-ops safely without
    # LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY, so trace_url is None there.
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        assert done["trace_url"]


@needs_openai_key
def test_citations_survive_a_thread_resume(monkeypatch, tmp_path):
    """Citations only ever exist as long as a turn is actively streaming
    unless they're persisted somewhere the checkpointer restores -- see
    agents/workflows/deterministic.py's finalize, which rides them along in
    the AIMessage's additional_kwargs for exactly this reason.

    Deliberately a different fixture (product name, numbers, filename) from
    test_full_pipeline_answers_with_citations_from_a_real_indexed_document's
    -- both tests index-then-delete a document against the same real,
    unisolated Postgres vector index, so identical fixture text let this
    test's retrieval intermittently pick up the *other* test's still-live
    document and cite its document_id instead, when both happened to run
    close together.
    """
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    with TestClient(app) as client:
        admin_email = _unique_email()
        monkeypatch.setenv("ADMIN_EMAILS", admin_email)
        admin_token = client.post(
            "/auth/signup", json={"email": admin_email, "password": "correct horse battery"}
        ).json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        content = (
            b"MIRA APEX Torque Spec\n\n"
            b"The MIRA APEX driver requires 2.4 Nm of torque when seating the "
            b"5.5mm compression screw. Do not exceed 2.8 Nm or the screw head may strip."
        )
        upload = client.post(
            "/documents/upload",
            headers=admin_headers,
            files={"file": ("mira-apex-torque-spec.txt", content, "text/plain")},
        )
        doc_id = upload.json()["id"]
        client.post(f"/documents/{doc_id}/index", headers=admin_headers)
        for _ in range(40):
            if client.get(f"/documents/{doc_id}", headers=admin_headers).json()["status"] == "done":
                break
            time.sleep(0.25)

        user_token = _user_token(client)
        user_headers = {"Authorization": f"Bearer {user_token}"}
        stream_res = client.post(
            "/chat/deterministic/stream",
            headers=user_headers,
            json={"message": "How much torque should I use on the MIRA APEX compression screw?"},
        )
        events = dict(_sse_events(stream_res.text))
        thread_id = events["thread"]["thread_id"]
        live_citations = events["done"]["citations"]

        transcript = client.get(f"/chat/threads/{thread_id}", headers=user_headers).json()

        client.delete(f"/documents/{doc_id}", headers=admin_headers)

    assert live_citations
    assistant_message = transcript["messages"][1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["citations"] == live_citations


def test_list_threads_requires_auth():
    with TestClient(app) as client:
        res = client.get("/chat/threads")
    assert res.status_code == 401


def test_list_threads_empty_for_a_new_user():
    with TestClient(app) as client:
        token = _user_token(client)
        res = client.get("/chat/threads", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json() == []


def test_get_thread_requires_auth():
    with TestClient(app) as client:
        res = client.get(f"/chat/threads/anyone:{uuid.uuid4().hex}")
    assert res.status_code == 401


def test_get_someone_elses_thread_id_is_rejected():
    with TestClient(app) as client:
        user_a = _signup(client)
        token_b = _user_token(client)
        # Correctly shaped (owner-prefixed) but belongs to user_a -- the
        # ownership check must reject this before ever touching the
        # checkpointer, so no real thread (or OPENAI_API_KEY) is needed.
        thread_id = f"{user_a['user']['id']}:{uuid.uuid4().hex}"
        res = client.get(
            f"/chat/threads/{thread_id}", headers={"Authorization": f"Bearer {token_b}"}
        )
    assert res.status_code == 403


def test_get_nonexistent_own_thread_404s():
    with TestClient(app) as client:
        user = _signup(client)
        thread_id = f"{user['user']['id']}:{uuid.uuid4().hex}"
        res = client.get(
            f"/chat/threads/{thread_id}",
            headers={"Authorization": f"Bearer {user['access_token']}"},
        )
    assert res.status_code == 404


@needs_openai_key
def test_thread_appears_in_sidebar_and_transcript_round_trips_both_turns(monkeypatch):
    # This test is about the sidebar/transcript round-trip, not self_eval's
    # judgment call -- a real judge's faithfulness/relevance score on these
    # two short, context-thin turns is exactly the kind of borderline case
    # request_clarification (2026-09-04) is *meant* to pause on, which would
    # otherwise make this test's pass rate depend on real LLM score
    # variance rather than the sidebar/transcript behavior it actually
    # exercises. Threshold set unreachably low so _should_clarify always
    # finalizes; test_ambiguous_query_pauses_for_clarification_and_
    # resume_completes_the_turn below covers the clarification pause itself.
    monkeypatch.setattr(det, "CLARIFICATION_SCORE_THRESHOLD", -1.0)
    with TestClient(app) as client:
        token = _user_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        # Names REFLEX HYBRID explicitly so detect_intent (ticket 09) resolves
        # confidently and doesn't pause the turn on a clarification -- this
        # test is about the sidebar/transcript round-trip, not intent
        # detection, so the message just needs to be unambiguous.
        first = client.post(
            "/chat/deterministic/stream",
            headers=headers,
            json={"message": "What torque should I use on a REFLEX HYBRID locking screw?"},
        )
        thread_id = _sse_events(first.text)[0][1]["thread_id"]

        listing = client.get("/chat/threads", headers=headers).json()
        assert len(listing) == 1
        assert listing[0]["thread_id"] == thread_id
        assert listing[0]["title"] == "What torque should I use on a REFLEX HYBRID locking screw?"

        second = client.post(
            "/chat/deterministic/stream",
            headers=headers,
            json={"message": "And in inch-pounds for the REFLEX HYBRID?", "thread_id": thread_id},
        )
        assert dict(_sse_events(second.text))["thread"]["thread_id"] == thread_id

        transcript = client.get(f"/chat/threads/{thread_id}", headers=headers).json()

    assert transcript["thread_id"] == thread_id
    roles = [m["role"] for m in transcript["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert (
        transcript["messages"][0]["content"]
        == "What torque should I use on a REFLEX HYBRID locking screw?"
    )
    assert transcript["messages"][2]["content"] == "And in inch-pounds for the REFLEX HYBRID?"


@needs_openai_key
def test_done_event_message_id_round_trips_through_feedback_and_transcript(monkeypatch):
    """Ticket 11: the "done" event's message_id is LangGraph's own
    auto-assigned uuid (add_messages) for the answer just finalized -- this
    proves it's real and stable enough to key feedback on, by submitting
    feedback against it and confirming GET /chat/threads/{id} (a completely
    separate read path, straight from the checkpointer) embeds that same
    feedback back onto the matching message.

    CLARIFICATION_SCORE_THRESHOLD forced unreachable: this turn has no real
    indexed document behind it, so a real self_eval score is unpredictable
    and this test isn't about that judgment call -- see the sidebar test's
    comment above for the same reasoning.
    """
    monkeypatch.setattr(det, "CLARIFICATION_SCORE_THRESHOLD", -1.0)
    with TestClient(app) as client:
        token = _user_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        res = client.post(
            "/chat/deterministic/stream",
            headers=headers,
            json={"message": "What torque should I use on a REFLEX HYBRID locking screw?"},
        )
        events = dict(_sse_events(res.text))
        thread_id = events["thread"]["thread_id"]
        message_id = events["done"]["message_id"]
        assert message_id

        feedback_res = client.post(
            "/feedback/",
            headers=headers,
            json={
                "thread_id": thread_id,
                "message_id": message_id,
                "flagged": True,
                "scores": {"faithfulness": 0.9, "relevance": 0.8, "style": 0.7, "citation": 1.0},
                "comment": "close but missing the inch-pounds conversion",
            },
        )
        assert feedback_res.status_code == 200

        transcript = client.get(f"/chat/threads/{thread_id}", headers=headers).json()

    assistant_message = next(m for m in transcript["messages"] if m["message_id"] == message_id)
    assert assistant_message["role"] == "assistant"
    assert assistant_message["feedback"]["flagged"] is True
    assert assistant_message["feedback"]["scores"]["faithfulness"] == 0.9
    assert assistant_message["feedback"]["comment"] == "close but missing the inch-pounds conversion"
    # The user's own turn never got feedback -- its embedded field stays null.
    user_message = next(m for m in transcript["messages"] if m["role"] == "user")
    assert user_message["feedback"] is None


@needs_openai_key
def test_ambiguous_query_pauses_for_clarification_and_resume_completes_the_turn(
    monkeypatch, tmp_path
):
    """Ticket 09's core acceptance criterion, end to end: a query that names
    no product system (matching evals/golden_datasets/intent_detection.jsonl's
    hand-authored expects_clarification: true fixtures --
    build_dataset.py's _AMBIGUOUS_FIXTURES, since the real feedback-notes.csv
    rows all confidently name one) pauses detect_intent's interrupt() rather
    than guessing, and answering it via POST /chat/resume resumes the same
    thread to a real "done" instead of starting a new turn.

    Creates and tags its own document (rather than relying on whatever
    systems already exist in this shared dev Postgres) so the clarification
    options this test asserts against are deterministic.

    CLARIFICATION_SCORE_THRESHOLD forced unreachable (2026-09-04): this
    test's uploaded document is placeholder text, so the resumed turn's
    real self_eval score would legitimately be terrible -- that's a second,
    different interrupt() (request_clarification's) this test isn't about,
    and would otherwise pause the resumed turn again instead of reaching
    "done".
    """
    monkeypatch.setattr(det, "CLARIFICATION_SCORE_THRESHOLD", -1.0)
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    with TestClient(app) as client:
        admin_email = _unique_email()
        monkeypatch.setenv("ADMIN_EMAILS", admin_email)
        admin_token = client.post(
            "/auth/signup", json={"email": admin_email, "password": "correct horse battery"}
        ).json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        system_name = f"ClarifySys-{uuid.uuid4().hex[:8]}"
        system_id = client.post(
            "/systems", headers=admin_headers, json={"name": system_name}
        ).json()["id"]
        upload = client.post(
            "/documents/upload",
            headers=admin_headers,
            data={"system_id": system_id},
            files={"file": ("clarify-doc.txt", b"placeholder content", "text/plain")},
        )
        doc_id = upload.json()["id"]

        user_headers = {"Authorization": f"Bearer {_user_token(client)}"}

        first = client.post(
            "/chat/deterministic/stream",
            headers=user_headers,
            json={"message": "What screws are in the set?"},
        )
        events = dict(_sse_events(first.text))
        assert "done" not in events  # paused, not finished
        clarification = events["clarification"]
        thread_id = clarification["thread_id"]
        assert clarification["question"]
        assert system_name in clarification["options"]

        second = client.post(
            "/chat/resume",
            headers=user_headers,
            json={"thread_id": thread_id, "human_input": system_name},
        )
        events2 = dict(_sse_events(second.text))
        assert events2["thread"]["thread_id"] == thread_id
        assert "done" in events2

        client.delete(f"/documents/{doc_id}", headers=admin_headers)


@needs_openai_key
def test_resume_without_a_pending_clarification_is_rejected(monkeypatch, tmp_path):
    """resume_chat's aget_state guard: resuming a thread that isn't actually
    paused on an interrupt (a fresh thread, or one that already finished)
    should 409 rather than silently no-op or error obscurely.

    CLARIFICATION_SCORE_THRESHOLD forced unreachable: a real self_eval
    score pausing the first turn on its own clarification interrupt would
    make this thread genuinely (if coincidentally) resumable, defeating the
    point of this test.
    """
    monkeypatch.setattr(det, "CLARIFICATION_SCORE_THRESHOLD", -1.0)
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    with TestClient(app) as client:
        user_headers = {"Authorization": f"Bearer {_user_token(client)}"}
        first = client.post(
            "/chat/deterministic/stream",
            headers=user_headers,
            json={"message": "How much torque should I use on the REFLEX HYBRID locking screw?"},
        )
        thread_id = dict(_sse_events(first.text))["thread"]["thread_id"]

        resume = client.post(
            "/chat/resume",
            headers=user_headers,
            json={"thread_id": thread_id, "human_input": "anything"},
        )
    assert resume.status_code == 409


@needs_openai_key
def test_a_long_first_message_is_truncated_into_the_thread_title():
    with TestClient(app) as client:
        token = _user_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        long_message = "How much torque should I use " + "really " * 20 + "on the locking screw?"

        client.post(
            "/chat/deterministic/stream", headers=headers, json={"message": long_message}
        )
        listing = client.get("/chat/threads", headers=headers).json()

    assert len(listing[0]["title"]) <= 61  # 60 chars + the "…" truncation marker
    assert listing[0]["title"].endswith("…")


def test_rerun_requires_admin():
    with TestClient(app) as client:
        token = _user_token(client)
        res = client.post(
            "/chat/rerun",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "original_thread_id": f"anyone:{uuid.uuid4().hex}",
                "original_message_id": str(uuid.uuid4()),
                "workflow_name": "deterministic",
            },
        )
    assert res.status_code == 403


def test_rerun_unknown_workflow_404s(monkeypatch):
    with TestClient(app) as client:
        admin_email = _unique_email()
        monkeypatch.setenv("ADMIN_EMAILS", admin_email)
        admin_token = client.post(
            "/auth/signup", json={"email": admin_email, "password": "correct horse battery"}
        ).json()["access_token"]
        res = client.post(
            "/chat/rerun",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "original_thread_id": f"anyone:{uuid.uuid4().hex}",
                "original_message_id": str(uuid.uuid4()),
                "workflow_name": "not-a-real-workflow",
            },
        )
    assert res.status_code == 404


@needs_openai_key
def test_rerun_missing_flagged_message_404s(monkeypatch):
    """The original thread is real, but the message_id doesn't exist in
    it -- a stale/mistyped id shouldn't 500."""
    with TestClient(app) as client:
        admin_email = _unique_email()
        monkeypatch.setenv("ADMIN_EMAILS", admin_email)
        admin_token = client.post(
            "/auth/signup", json={"email": admin_email, "password": "correct horse battery"}
        ).json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        original = client.post(
            "/chat/deterministic/stream",
            headers=admin_headers,
            json={"message": "What torque should I use on a REFLEX HYBRID locking screw?"},
        )
        original_thread_id = dict(_sse_events(original.text))["thread"]["thread_id"]

        res = client.post(
            "/chat/rerun",
            headers=admin_headers,
            json={
                "original_thread_id": original_thread_id,
                "original_message_id": str(uuid.uuid4()),
                "workflow_name": "deterministic",
            },
        )
    assert res.status_code == 404


@needs_openai_key
def test_rerun_replays_history_and_stays_out_of_the_admins_own_sidebar(monkeypatch):
    """Ticket 15's core loop, end to end: a real 2-turn conversation, the
    second answer flagged, then an admin reruns it. Asserts the rerun (a)
    actually happened (a "done" event came back), (b) replayed the first
    turn's real content into the new thread rather than starting blank, (c)
    never touches the original rep's thread, (d) doesn't show up in the
    admin's own GET /chat/threads sidebar despite being a real chat_threads
    row, and (e) is listed by GET /feedback/{message_id}/reruns.

    CLARIFICATION_SCORE_THRESHOLD forced unreachable for the same reason as
    the sidebar/feedback tests above -- this test is about the rerun
    mechanism, not self_eval's judgment call on these specific questions.
    """
    monkeypatch.setattr(det, "CLARIFICATION_SCORE_THRESHOLD", -1.0)
    with TestClient(app) as client:
        admin_email = _unique_email()
        monkeypatch.setenv("ADMIN_EMAILS", admin_email)
        admin_token = client.post(
            "/auth/signup", json={"email": admin_email, "password": "correct horse battery"}
        ).json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        rep_token = _user_token(client)
        rep_headers = {"Authorization": f"Bearer {rep_token}"}

        first_message = "What torque should I use on a REFLEX HYBRID locking screw?"
        first = client.post(
            "/chat/deterministic/stream", headers=rep_headers, json={"message": first_message}
        )
        original_thread_id = dict(_sse_events(first.text))["thread"]["thread_id"]

        second = client.post(
            "/chat/deterministic/stream",
            headers=rep_headers,
            json={"message": "And in inch-pounds?", "thread_id": original_thread_id},
        )
        flagged_message_id = dict(_sse_events(second.text))["done"]["message_id"]

        flag = client.post(
            "/feedback/",
            headers=rep_headers,
            json={
                "thread_id": original_thread_id,
                "message_id": flagged_message_id,
                "flagged": True,
                "scores": {},
            },
        )
        assert flag.status_code == 200

        rerun = client.post(
            "/chat/rerun",
            headers=admin_headers,
            json={
                "original_thread_id": original_thread_id,
                "original_message_id": flagged_message_id,
                "workflow_name": "deterministic",
            },
        )
        rerun_events = dict(_sse_events(rerun.text))
        assert "done" in rerun_events
        rerun_thread_id = rerun_events["thread"]["thread_id"]
        assert rerun_thread_id != original_thread_id

        rerun_transcript = client.get(
            f"/chat/threads/{rerun_thread_id}", headers=admin_headers
        ).json()
        admin_sidebar = client.get("/chat/threads", headers=admin_headers).json()
        reruns_for_message = client.get(
            f"/feedback/{flagged_message_id}/reruns", headers=admin_headers
        ).json()

        # The original rep's own thread must be completely untouched.
        original_transcript = client.get(
            f"/chat/threads/{original_thread_id}", headers=rep_headers
        ).json()

    assert original_transcript["messages"][0]["content"] == first_message
    assert len(original_transcript["messages"]) == 4  # unchanged by the rerun

    roles = [m["role"] for m in rerun_transcript["messages"]]
    contents = [m["content"] for m in rerun_transcript["messages"]]
    # Replayed history (first turn) followed by a freshly generated answer
    # to the *second* turn's question -- not a blank-context rerun.
    assert roles[0] == "user" and contents[0] == first_message
    assert roles[-2] == "user" and contents[-2] == "And in inch-pounds?"
    assert roles[-1] == "assistant"

    assert rerun_thread_id not in [t["thread_id"] for t in admin_sidebar]
    assert [r["thread_id"] for r in reruns_for_message] == [rerun_thread_id]
    assert reruns_for_message[0]["workflow_name"] == "deterministic"

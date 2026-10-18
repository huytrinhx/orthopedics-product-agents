"""Issue #24: per-slot models. The probe, the output fallback, the 30-second
settings cache, the Settings routes, and the chat request's admin-only
overrides -- all against fake models and a fake OpenRouter catalog, so CI
needs no OpenRouter or OpenAI key. The routes and chat turns run against
the real app and real Postgres, like test_admin_settings.py.
"""
import argparse
import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agents.workflows.deterministic as det
import agents.workflows.react_agent as react
from api.main import app
from auth.repository import set_user_active
from evals import harness
from model_assignments import catalog, probe, repository, runtime
from model_assignments.slots import CHAT_SLOTS, SLOT_DEFINITIONS, Need, Slot

# What a parsing fake answers for each structured-output schema the probe
# and the nodes use.
_PARSED_OUTPUTS = {
    "_IntentClassification": {"product_family": None, "question_type": ["unclassified"]},
    "_QuestionTypeReply": {"question_type": "technique_procedural"},
    "_JudgeScores": {"faithfulness": 1, "relevance": 1, "style": 1, "citation": 1},
    "_ExtractionOutput": {"differentiations": [], "procedure_requirements": []},
    "_TrayLayoutExtraction": {"groups": []},
}


class _FakeRunnable:
    def __init__(self, invoke):
        self._invoke = invoke

    async def ainvoke(self, messages):
        return self._invoke(messages)


class _FakeModel:
    """Stands in for a chat model through every call shape the slots use:
    `with_structured_output`, `bind_tools`, plain `ainvoke`, and `astream`.
    A parsing fake answers each the way a compatible model would; one that
    doesn't parse fails each the way an incompatible one does.
    """

    def __init__(self, parses: bool, answer: str = "A cannulated screw goes over a guidewire."):
        self.parses = parses
        self.answer = answer
        self.prompts: list = []

    def with_structured_output(self, schema):
        def invoke(messages):
            self.prompts.append(messages)
            if not self.parses:
                raise OutputParserException("Invalid json output: I'd rather chat.")
            return schema.model_validate(_PARSED_OUTPUTS[schema.__name__])

        return _FakeRunnable(invoke)

    def bind_tools(self, tools):
        def invoke(messages):
            self.prompts.append(messages)
            if not self.parses:
                return AIMessage(content="I don't use tools.")
            if any(getattr(message, "type", None) == "tool" for message in messages):
                return AIMessage(content="MPPA100L is full thread.")
            return AIMessage(
                content="",
                tool_calls=[{"name": "part_lookup", "args": {"term": "MPPA100L"}, "id": "call-1"}],
            )

        return _FakeRunnable(invoke)

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content=self.answer if self.parses else "")

    async def astream(self, messages):
        self.prompts.append(messages)
        for word in (self.answer.split(" ") if self.parses else ["", ""]):
            yield AIMessageChunk(content=f"{word} " if word else "")


def _run_isolated(coroutine):
    """Runs a coroutine to completion on its own loop in its own thread, so
    a sync fixture can touch the database without disturbing the session
    loop the async tests share.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


@pytest.fixture(autouse=True)
def _clean_assignments():
    """Every test starts and ends with an empty table and a cold cache, so
    nothing saved here leaks into another test's turn.
    """

    async def clear() -> None:
        for slot in Slot:
            await repository.delete_assignment(slot.value)

    _run_isolated(clear())
    catalog.clear_catalog_cache()
    yield
    _run_isolated(clear())
    catalog.clear_catalog_cache()


# ---------------------------------------------------------------- probe


@pytest.mark.parametrize("slot", list(Slot))
async def test_every_slot_probe_passes_on_a_model_that_parses(monkeypatch, slot):
    fake_model = _FakeModel(parses=True)
    monkeypatch.setattr(probe, "build_probe_model", lambda model_id, reasoning_effort: fake_model)

    result = await probe.probe_slot(slot, "vendor/model", "low")

    assert result.passed, result.details
    assert len(result.details) == len(probe.SLOT_CHECKS[slot])
    assert all(detail.startswith("PASS ") for detail in result.details)


@pytest.mark.parametrize("slot", list(Slot))
async def test_every_slot_probe_fails_with_a_reason_on_a_model_that_doesnt(monkeypatch, slot):
    monkeypatch.setattr(
        probe, "build_probe_model", lambda model_id, reasoning_effort: _FakeModel(parses=False)
    )

    result = await probe.probe_slot(slot, "vendor/model", None)

    assert not result.passed
    assert all(detail.startswith("FAIL ") for detail in result.details)
    # The reason is what Settings shows: the exception, or what was missing.
    assert any(
        "OutputParserException" in detail or "no tool call" in detail or "no text" in detail
        for detail in result.details
    )


async def test_the_tray_layout_probe_sends_the_image_fixture(monkeypatch):
    fake_model = _FakeModel(parses=True)
    monkeypatch.setattr(probe, "build_probe_model", lambda model_id, reasoning_effort: fake_model)

    await probe.probe_slot(Slot.TRAY_LAYOUT_EXTRACTION, "vendor/model", None)

    user_content = fake_model.prompts[0][1][1]
    image_part = next(part for part in user_content if part["type"] == "image_url")
    assert image_part["image_url"]["url"] == f"data:image/png;base64,{probe.tray_image_b64()}"
    assert probe.TRAY_IMAGE_FIXTURE.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------- output fallback


def _recording_get_chat_model(models_by_arguments: dict, calls: list):
    def fake_get_chat_model(*arguments):
        calls.append(arguments)
        return models_by_arguments[arguments]

    return fake_get_chat_model


async def test_an_assigned_slots_unparseable_output_retries_once_on_chat_model(monkeypatch, caplog):
    monkeypatch.setattr(runtime.logger, "disabled", False)
    calls: list = []
    get_chat_model = _recording_get_chat_model(
        {("vendor/model", "low"): _FakeModel(parses=False), (): _FakeModel(parses=True)}, calls
    )
    slot_model = runtime.SlotModel(Slot.JUDGE, "vendor/model", "low", "setting")

    with caplog.at_level(logging.WARNING, logger=runtime.__name__):
        result = await runtime.invoke_with_output_fallback(
            slot_model,
            get_chat_model,
            lambda model: model.with_structured_output(det._QuestionTypeReply).ainvoke([]),
        )

    assert result.question_type == "technique_procedural"
    assert calls == [("vendor/model", "low"), ()]
    [record] = [record for record in caplog.records if record.message.startswith("slot_output_fallback")]
    assert record.slot == "judge"
    assert record.model == "vendor/model"
    assert "OutputParserException" in record.reason


async def test_an_unusable_answer_also_falls_back(monkeypatch):
    calls: list = []
    get_chat_model = _recording_get_chat_model(
        {("vendor/model", None): _FakeModel(parses=False), (): _FakeModel(parses=True)}, calls
    )
    slot_model = runtime.SlotModel(Slot.DETERMINISTIC_ANSWER, "vendor/model", None, "override")

    response = await runtime.invoke_with_output_fallback(
        slot_model,
        get_chat_model,
        lambda model: model.ainvoke([]),
        is_usable=lambda candidate: bool(candidate.text.strip()),
    )

    assert response.text
    assert calls == [("vendor/model", None), ()]


async def test_a_default_slot_never_retries_and_fails_as_before():
    calls: list = []
    get_chat_model = _recording_get_chat_model({(): _FakeModel(parses=False)}, calls)

    with pytest.raises(OutputParserException):
        await runtime.invoke_with_output_fallback(
            runtime.SlotModel(Slot.JUDGE),
            get_chat_model,
            lambda model: model.with_structured_output(det._QuestionTypeReply).ainvoke([]),
        )
    assert calls == [()]


async def test_the_intent_classifier_runs_on_the_slot_model_it_is_given(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        det,
        "get_chat_model",
        _recording_get_chat_model({("vendor/model", "none"): _FakeModel(parses=True)}, calls),
    )

    result = await det._classify_intent(
        "What screws?",
        [],
        slot_model=runtime.SlotModel(Slot.INTENT_CLASSIFICATION, "vendor/model", "none", "override"),
    )

    assert result.question_type == ["unclassified"]
    assert calls == [("vendor/model", "none")]


async def test_a_react_turn_reads_its_slot_from_the_run_config_and_falls_back_on_a_bad_tool_call(
    monkeypatch,
):
    """Inside a graph run, a node finds its slot in the run config (what
    chat.py puts there). The assigned model's malformed tool call is retried
    on CHAT_MODEL, which answers.
    """
    class _Scripted:
        def __init__(self, reply):
            self.reply = reply
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            return self.reply

    assigned = _Scripted(
        AIMessage(
            content="",
            invalid_tool_calls=[{"name": "part_lookup", "args": "{not json", "id": "call-1", "error": None}],
        )
    )
    default = _Scripted(AIMessage(content="No SKU was named, so there's nothing to look up."))
    calls: list = []
    monkeypatch.setattr(
        react,
        "get_chat_model",
        _recording_get_chat_model({("vendor/agent", "medium"): assigned, (): default}, calls),
    )

    async def fake_detect_intent(state):
        return {"resolved_family": None, "resolved_family_id": None, "detected_question_types": ["unclassified"]}

    monkeypatch.setattr(react, "detect_intent", fake_detect_intent)
    turn_models = {
        Slot.REACT_AGENT.value: runtime.SlotModel(Slot.REACT_AGENT, "vendor/agent", "medium", "override")
    }
    graph = react.build_graph(InMemorySaver())
    thread_id = f"test-{uuid.uuid4().hex[:10]}"
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="hello")],
            "query": "hello",
            "search_query": "hello",
            "correction_rounds": 0,
            "fact_check_issues": [],
            "user_id": "u1",
            "thread_id": thread_id,
        },
        {"configurable": {"thread_id": thread_id, **runtime.slot_models_config(turn_models)}},
    )

    assert calls == [("vendor/agent", "medium"), ()]
    assert result["answer"] == "No SKU was named, so there's nothing to look up."


# ---------------------------------------------------------------- cache


async def test_a_saved_slot_is_read_through_a_30_second_cache(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(repository, "_clock", lambda: now[0])
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    repository.invalidate_cache()

    assert (await runtime.configured_slot_model(Slot.JUDGE)).is_default

    # Written behind the cache's back, the way another process would.
    connection = await repository.get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO model_assignments (slot, model_id, reasoning_effort) VALUES (%s, %s, %s)",
                ("judge", "vendor/judge-model", "low"),
            )
        await connection.commit()
    finally:
        await connection.close()

    now[0] += 29
    assert (await runtime.configured_slot_model(Slot.JUDGE)).is_default
    now[0] += 1
    judge = await runtime.configured_slot_model(Slot.JUDGE)
    assert (judge.model_id, judge.reasoning_effort, judge.source) == ("vendor/judge-model", "low", "setting")


async def test_without_the_gateway_every_slot_runs_the_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    await repository.save_assignment("judge", "vendor/judge-model", "low", None)

    assert (await runtime.configured_slot_model(Slot.JUDGE)).is_default
    turn_models = await runtime.resolve_turn_slot_models(
        {"react_agent": {"model_id": "vendor/agent", "reasoning_effort": "high"}}
    )
    assert all(slot_model.is_default for slot_model in turn_models.values())


# ---------------------------------------------------------------- Settings routes

_CATALOG = [
    {
        "id": "vendor/structured-model",
        "name": "Structured Model",
        "pricing": {"prompt": "0.000001", "completion": "0.000004"},
        "architecture": {"input_modalities": ["text"]},
        "supported_parameters": ["structured_outputs", "reasoning"],
    },
    {
        "id": "vendor/vision-model",
        "name": "Vision Model",
        "pricing": {"prompt": "0.000002", "completion": "0.00001"},
        "architecture": {"input_modalities": ["text", "image"]},
        "supported_parameters": ["structured_outputs", "tools"],
    },
    {
        "id": "vendor/vision-model-2026-09-01",
        "name": "Vision Model (2026-09-01)",
        "pricing": {"prompt": "0.000002", "completion": "0.00001"},
        "architecture": {"input_modalities": ["text", "image"]},
        "supported_parameters": ["structured_outputs", "tools"],
    },
    {
        "id": "vendor/plain-model",
        "name": "Plain Model",
        "pricing": {"prompt": "-1", "completion": "-1"},
        "architecture": {"input_modalities": ["text"]},
        "supported_parameters": [],
    },
]


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _admin_headers(client: TestClient, monkeypatch) -> dict:
    email = _unique_email()
    monkeypatch.setenv("ADMIN_EMAILS", email)
    response = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _user_headers(client: TestClient) -> dict:
    response = client.post(
        "/auth/signup", json={"email": _unique_email(), "password": "correct horse battery"}
    )
    body = response.json()
    _run_isolated(set_user_active(uuid.UUID(body["user"]["id"]), True))
    return {"Authorization": f"Bearer {body['access_token']}"}


@pytest.fixture
def gateway(monkeypatch):
    """OPENROUTER_API_KEY set, the catalog faked, and the probe on a model
    that parses -- `gateway.parses = False` makes it one that doesn't.
    """

    class Gateway:
        parses = True

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    async def fake_fetch_openrouter_models():
        return _CATALOG

    monkeypatch.setattr(catalog, "fetch_openrouter_models", fake_fetch_openrouter_models)
    monkeypatch.setattr(
        probe, "build_probe_model", lambda model_id, reasoning_effort: _FakeModel(Gateway.parses)
    )
    return Gateway


def test_the_settings_routes_are_admin_only():
    with TestClient(app) as client:
        headers = _user_headers(client)
        assert client.get("/admin/model-assignments", headers=headers).status_code == 403
        assert (
            client.put(
                "/admin/model-assignments/judge",
                headers=headers,
                json={"model_id": "vendor/structured-model"},
            ).status_code
            == 403
        )


def test_without_the_gateway_settings_are_read_only_and_show_the_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _run_isolated(repository.save_assignment("judge", "vendor/structured-model", "low", None))
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        body = client.get("/admin/model-assignments", headers=headers).json()
        save = client.put(
            "/admin/model-assignments/judge", headers=headers, json={"model_id": "vendor/structured-model"}
        )

    assert body["gateway_configured"] is False
    assert [slot["slot"] for slot in body["slots"]] == [slot.value for slot in Slot]
    assert all(slot["assignment"] is None for slot in body["slots"])
    assert save.status_code == 409


def test_every_slot_declares_its_needs_and_the_judge_warns(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        slots = client.get("/admin/model-assignments", headers=headers).json()["slots"]
    with_needs = {slot["slot"]: slot for slot in slots}
    assert with_needs["tray_layout_extraction"]["needs"] == ["structured_output", "image_input"]
    assert with_needs["react_agent"]["needs"] == ["tool_calling"]
    assert with_needs["deterministic_answer"]["needs"] == ["streamed_text"]
    assert "score comparison" in with_needs["judge"]["warning"]
    assert {slot: definition.default_reasoning_effort for slot, definition in SLOT_DEFINITIONS.items()} == {
        Slot.INTENT_CLASSIFICATION: "none",
        Slot.DETERMINISTIC_ANSWER: "low",
        Slot.REACT_AGENT: "medium",
        Slot.JUDGE: "low",
        Slot.ENTITY_EXTRACTION: "low",
        Slot.TRAY_LAYOUT_EXTRACTION: "low",
    }


def test_the_picker_lists_only_models_covering_the_slots_needs(gateway, monkeypatch):
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        tray_models = client.get("/admin/model-assignments/tray_layout_extraction/models", headers=headers)
        judge_models = client.get("/admin/model-assignments/judge/models", headers=headers)
        answer_models = client.get("/admin/model-assignments/deterministic_answer/models", headers=headers)

    # Image input: only the vision model, by its dated id (the alias is hidden).
    assert [model["model_id"] for model in tray_models.json()] == ["vendor/vision-model-2026-09-01"]
    assert [model["model_id"] for model in judge_models.json()] == [
        "vendor/structured-model",
        "vendor/vision-model-2026-09-01",
    ]
    structured = judge_models.json()[0]
    assert structured["prompt_price_per_million"] == 1.0
    assert structured["completion_price_per_million"] == 4.0
    assert structured["supports_reasoning"] is True
    # Streamed text needs nothing a chat model lacks; unpriced is None.
    plain = next(model for model in answer_models.json() if model["model_id"] == "vendor/plain-model")
    assert plain["prompt_price_per_million"] is None


def test_an_alias_is_hidden_only_by_its_own_dated_id():
    def model(model_id: str) -> catalog.CatalogModel:
        return catalog.CatalogModel(model_id, model_id, None, None, frozenset(), frozenset())

    listed = [
        model("openai/gpt-4o"),
        model("openai/gpt-4o-mini-2024-07-18"),
        model("mistralai/mistral-large"),
        model("mistralai/mistral-large-2411"),
        model("anthropic/claude-3.5-sonnet-20240620"),
    ]

    assert [kept.model_id for kept in catalog.prefer_dated_ids(listed)] == [
        "openai/gpt-4o",
        "openai/gpt-4o-mini-2024-07-18",
        "mistralai/mistral-large-2411",
        "anthropic/claude-3.5-sonnet-20240620",
    ]


def test_saving_a_model_that_passes_its_probe_records_who_and_when(gateway, monkeypatch):
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        response = client.put(
            "/admin/model-assignments/judge",
            headers=headers,
            json={"model_id": "vendor/structured-model", "reasoning_effort": "low"},
        )
        listed = client.get("/admin/model-assignments", headers=headers).json()

    assert response.status_code == 200
    judge = next(slot for slot in listed["slots"] if slot["slot"] == "judge")
    assert judge["assignment"]["model_id"] == "vendor/structured-model"
    assert judge["assignment"]["reasoning_effort"] == "low"
    assert judge["assignment"]["updated_by_email"].endswith("@example.com")
    assert judge["assignment"]["updated_at"]


def test_saving_a_model_that_fails_its_probe_is_rejected_with_the_reason(gateway, monkeypatch):
    gateway.parses = False
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        response = client.put(
            "/admin/model-assignments/judge", headers=headers, json={"model_id": "vendor/structured-model"}
        )
        listed = client.get("/admin/model-assignments", headers=headers).json()

    assert response.status_code == 422
    assert "failed the Judge probe" in response.json()["detail"]
    assert "OutputParserException" in response.json()["detail"]
    assert next(slot for slot in listed["slots"] if slot["slot"] == "judge")["assignment"] is None


def test_test_runs_the_probe_without_saving(gateway, monkeypatch):
    gateway.parses = False
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        response = client.post(
            "/admin/model-assignments/react_agent/test",
            headers=headers,
            json={"model_id": "vendor/vision-model"},
        )
        listed = client.get("/admin/model-assignments", headers=headers).json()

    assert response.status_code == 200
    assert response.json()["passed"] is False
    assert "no tool call" in response.json()["details"][0]
    assert all(slot["assignment"] is None for slot in listed["slots"])


@pytest.mark.parametrize(
    ("slot_name", "body", "expected_status"),
    [
        # Catalog metadata doesn't cover image input.
        ("tray_layout_extraction", {"model_id": "vendor/structured-model"}, 400),
        # No reasoning support, so no effort.
        ("judge", {"model_id": "vendor/vision-model", "reasoning_effort": "high"}, 400),
        ("judge", {"model_id": "vendor/not-in-the-catalog"}, 400),
        ("not_a_slot", {"model_id": "vendor/structured-model"}, 404),
        ("judge", {"model_id": "vendor/structured-model", "reasoning_effort": "extreme"}, 422),
    ],
)
def test_saving_is_refused_before_the_probe_when_the_catalog_rules_it_out(
    gateway, monkeypatch, slot_name, body, expected_status
):
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        response = client.put(f"/admin/model-assignments/{slot_name}", headers=headers, json=body)
    assert response.status_code == expected_status


def test_reset_puts_a_slot_back_on_the_default(gateway, monkeypatch):
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        client.put("/admin/model-assignments/judge", headers=headers, json={"model_id": "vendor/structured-model"})
        response = client.delete("/admin/model-assignments/judge", headers=headers)

    assert response.status_code == 200
    assert all(slot["assignment"] is None for slot in response.json()["slots"])


# ---------------------------------------------------------------- chat turns


def _sse_events(text: str) -> dict:
    events = {}
    event_name = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event_name = line[len("event: ") :]
        elif line.startswith("data: "):
            events[event_name] = json.loads(line[len("data: ") :])
    return events


def _fake_deterministic_turn(monkeypatch) -> list:
    """Runs the real detect_intent and generate on a recording
    get_chat_model, with every node in between faked. Returns the list of
    get_chat_model arguments each model call asked for.
    """
    calls: list = []

    def fake_get_chat_model(*arguments):
        calls.append(arguments)
        return _FakeModel(parses=True, answer="The answer.")

    async def no_families():
        return []

    async def fake_resolve_synonyms(state):
        return {"resolved_canonical_terms": [], "synonym_ambiguity": None}

    async def fake_hybrid_retrieve(state):
        return {"retrieved": []}

    async def fake_resolve_query_parts(state):
        return {"query_parts": [], "turn_tray_ids": []}

    async def fake_rerank(state):
        return {"reranked": []}

    async def fake_resolve_skus(state):
        return {"resolved_parts": []}

    async def fake_aggregate_facts(state):
        return {"aggregated_facts": ""}

    async def fake_self_eval(state):
        return {"fact_check_issues": []}

    monkeypatch.setattr(det, "get_chat_model", fake_get_chat_model)
    monkeypatch.setattr(det, "list_families_with_trays", no_families)
    monkeypatch.setattr(det, "resolve_synonyms", fake_resolve_synonyms)
    monkeypatch.setattr(det, "hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr(det, "resolve_query_parts", fake_resolve_query_parts)
    monkeypatch.setattr(det, "rerank", fake_rerank)
    monkeypatch.setattr(det, "resolve_skus", fake_resolve_skus)
    monkeypatch.setattr(det, "aggregate_facts", fake_aggregate_facts)
    monkeypatch.setattr(det, "self_eval", fake_self_eval)
    return calls


_OVERRIDES = {
    "intent_classification": {"model_id": "vendor/fast-classifier", "reasoning_effort": "none"},
    "deterministic_answer": {"model_id": "vendor/writer", "reasoning_effort": None},
}


def test_an_empty_table_runs_every_slot_on_the_default_call(monkeypatch):
    calls = _fake_deterministic_turn(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/chat/deterministic/stream", headers=_user_headers(client), json={"message": "What screws?"}
        )

    done = _sse_events(response.text)["done"]
    assert done["answer"] == "The answer."
    assert "models" not in done
    assert calls == [(), ()]


def test_a_saved_slot_takes_effect_on_the_next_turn(gateway, monkeypatch):
    calls = _fake_deterministic_turn(monkeypatch)
    _run_isolated(repository.save_assignment("deterministic_answer", "vendor/writer", "low", None))
    with TestClient(app) as client:
        response = client.post(
            "/chat/deterministic/stream", headers=_user_headers(client), json={"message": "What screws?"}
        )

    assert "models" not in _sse_events(response.text)["done"]
    # Intent classification is unset, so it still makes the default call.
    assert calls == [(), ("vendor/writer", "low")]


def test_a_non_admin_sending_overrides_is_rejected(gateway):
    with TestClient(app) as client:
        response = client.post(
            "/chat/stream",
            headers=_user_headers(client),
            json={"message": "What screws?", "model_overrides": _OVERRIDES},
        )
    assert response.status_code == 403


def test_overrides_must_name_chat_slots(gateway, monkeypatch):
    with TestClient(app) as client:
        response = client.post(
            "/chat/deterministic/stream",
            headers=_admin_headers(client, monkeypatch),
            json={"message": "What screws?", "model_overrides": {"judge": {"model_id": "vendor/x"}}},
        )
    assert response.status_code == 400
    assert {slot.value for slot in CHAT_SLOTS} == {
        "intent_classification",
        "deterministic_answer",
        "react_agent",
    }


def test_a_rerun_with_overrides_runs_on_them_shows_them_and_keeps_them_for_the_conversation(
    gateway, monkeypatch
):
    calls = _fake_deterministic_turn(monkeypatch)
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        first = _sse_events(
            client.post(
                "/chat/deterministic/stream",
                headers=headers,
                json={"message": "What screws?", "model_overrides": _OVERRIDES},
            ).text
        )
        follow_up = _sse_events(
            client.post(
                "/chat/deterministic/stream",
                headers=headers,
                json={"message": "And the wires?", "thread_id": first["thread"]["thread_id"]},
            ).text
        )
        fresh = _sse_events(
            client.post("/chat/deterministic/stream", headers=headers, json={"message": "Fresh?"}).text
        )

    expected_calls = [("vendor/fast-classifier", "none"), ("vendor/writer", None)]
    assert calls == [*expected_calls, *expected_calls, (), ()]
    shown = first["done"]["models"]
    assert shown["intent_classification"] == {
        "model_id": "vendor/fast-classifier",
        "reasoning_effort": "none",
        "source": "override",
    }
    assert shown["react_agent"]["source"] == "default"
    assert follow_up["done"]["models"] == shown
    # Overrides belong to that one conversation, not to the admin.
    assert "models" not in fresh["done"]


def test_the_turn_log_records_each_slots_model(gateway, monkeypatch, caplog):
    monkeypatch.setattr(runtime.logger, "disabled", False)
    _fake_deterministic_turn(monkeypatch)
    with caplog.at_level(logging.INFO, logger=runtime.__name__), TestClient(app) as client:
        client.post(
            "/chat/deterministic/stream",
            headers=_admin_headers(client, monkeypatch),
            json={"message": "What screws?", "model_overrides": _OVERRIDES},
        )

    [record] = [record for record in caplog.records if record.message.startswith("chat_turn_models")]
    assert record.workflow == "deterministic"
    assert record.slot_models["deterministic_answer"]["model_id"] == "vendor/writer"
    assert "intent_classification=vendor/fast-classifier@none" in record.message


# ---------------------------------------------------------------- harness


def test_the_harness_parses_model_overrides_by_slot():
    assert harness.parse_model_override("intent_classification=vendor/model@low") == (
        Slot.INTENT_CLASSIFICATION,
        "vendor/model",
        "low",
    )
    assert harness.parse_model_override("judge=openai/gpt-5-mini") == (Slot.JUDGE, "openai/gpt-5-mini", None)
    for bad_value in ("judge", "nope=vendor/model", "judge=vendor/model@extreme", "judge="):
        with pytest.raises(argparse.ArgumentTypeError):
            harness.parse_model_override(bad_value)


async def test_the_harness_report_records_every_slots_model(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    slot_models = await harness.resolve_eval_slot_models(
        [(Slot.INTENT_CLASSIFICATION, "openai/gpt-5-mini", "low")]
    )

    report = harness.format_slot_models(slot_models)
    assert "intent_classification: openai/gpt-5-mini (reasoning effort: low; override)" in report
    for slot in Slot:
        assert f"  {slot.value}: " in report


def test_need_values_are_what_the_ui_shows():
    assert [need.value for need in Need] == [
        "structured_output",
        "streamed_text",
        "tool_calling",
        "image_input",
    ]

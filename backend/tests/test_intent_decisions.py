"""Issue #23: intent classification on a decision model. The Decisions API
client against a mocked HTTP transport, the Choice -> classification rules,
the workflow's decision path and its fallback to the chat path, the
decision probe, the Settings routes for a decision model, the chat
override check, and the harness's `--model` -- all with fakes, so CI needs
no OpenRouter key and nothing here spends money.
"""
import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

import agents.workflows.deterministic as det
from agents import intent_decisions
from agents.question_types import QUESTION_TYPE_NAMES
from api.main import app
from catalog.repository import ProductFamilyRecord
from config.llm_clients import (
    OPENROUTER_DECISIONS_URL,
    ChoiceAnswer,
    DecisionModelError,
    DecisionResponse,
    DecisionUsage,
    OpenRouterDecisionClient,
    collect_decision_usage,
    record_decision_usage,
)
from evals import harness
from evals.intent_eval import evaluate_row
from model_assignments import catalog, probe, repository, runtime
from model_assignments.slots import (
    DEFAULT_FAMILY_CONFIDENCE_THRESHOLD,
    DEFAULT_QUESTION_TYPE_MARGIN,
    Slot,
)

_FAMILY_NAMES = ["MIS - Foot Recon System", "REFLEX HYBRID", "REFLEX Foot Recon"]
_DECISION_MODEL = runtime.decision_slot_model(
    Slot.INTENT_CLASSIFICATION, "typesafe/jev-1.13-20260917", "override"
)


class _FakeDecisionClient:
    """Answers each question id with a canned (choice, probabilities),
    recording every request.
    """

    def __init__(self, answers: dict[str, tuple[str, dict[str, float]]], cost: float = 0.0):
        self.answers = answers
        self.cost = cost
        self.requests: list[dict] = []

    async def decide(self, model_id: str, state: dict, questions: dict) -> DecisionResponse:
        self.requests.append({"model": model_id, "state": state, "questions": questions})
        return DecisionResponse(
            model=model_id,
            answers={
                question_id: ChoiceAnswer(*self.answers[question_id]) for question_id in questions
            },
            usage=DecisionUsage(input_tokens=120, output_tokens=0, cost=self.cost),
        )


def _intent_answers(
    family: tuple[str, dict[str, float]],
    question_type: tuple[str, dict[str, float]] = ("product_characteristics", {"product_characteristics": 0.9}),
) -> dict:
    return {"product_family": family, "question_type": question_type}


def _no_chat_model(*arguments):
    raise AssertionError(f"the chat model was called with {arguments}")


# ---------------------------------------------------------------- client


def _transport(handler):
    return httpx.MockTransport(handler)


_CHOICE_QUESTION = {
    "pick": {"type": "choice", "instructions": "Pick one.", "criteria": {"red": "Red.", "blue": "Blue."}}
}


async def test_the_client_posts_to_openrouters_decisions_api_and_parses_the_choice():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "dec-1",
                "model": "typesafe/jev-1.13-20260917",
                "answers": {
                    "pick": {
                        "type": "choice",
                        "choice": "blue",
                        "probabilities": {"red": 0.2, "blue": 0.8},
                        "confidence": 0.8,
                    }
                },
                "usage": {"input_tokens": 40, "output_tokens": 0, "cost": 0.000002},
            },
        )

    client = OpenRouterDecisionClient("test-key", transport=_transport(handler))
    response = await client.decide("typesafe/jev-1.13-20260917", {"text": "sky"}, _CHOICE_QUESTION)

    [request] = seen
    assert str(request.url) == OPENROUTER_DECISIONS_URL
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "typesafe/jev-1.13-20260917",
        "state": {"text": "sky"},
        "questions": _CHOICE_QUESTION,
    }
    assert response.answers["pick"] == ChoiceAnswer("blue", {"red": 0.2, "blue": 0.8})
    assert response.usage == DecisionUsage(input_tokens=40, output_tokens=0, cost=0.000002)
    # Only what a node acts on is counted (intent_decisions), not every call.
    with collect_decision_usage() as collected:
        record_decision_usage(response)
    assert collected == {"typesafe/jev-1.13-20260917": [response.usage]}


async def test_the_client_falls_back_to_the_confidence_when_probabilities_are_omitted():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "m",
                "answers": {"pick": {"type": "choice", "choice": "red", "confidence": 0.7}},
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        )

    client = OpenRouterDecisionClient("test-key", transport=_transport(handler))
    response = await client.decide("m", "sky", _CHOICE_QUESTION)

    assert response.answers["pick"].probabilities == {"red": 0.7}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, text="overloaded"),
        httpx.Response(402, json={"error": "out of credit"}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"model": "m", "answers": {}}),
        # Not one of the question's own options.
        httpx.Response(
            200,
            json={"model": "m", "answers": {"pick": {"type": "choice", "choice": "green", "confidence": 1}}},
        ),
        httpx.Response(200, json={"model": "m", "answers": {"pick": {"type": "noul", "noul": 0.4}}}),
        httpx.Response(200, json={"model": "m", "answers": {"pick": {"type": "choice", "choice": "red"}}}),
    ],
)
async def test_any_unusable_decision_response_raises_one_error(response):
    client = OpenRouterDecisionClient("test-key", transport=_transport(lambda request: response))
    with pytest.raises(DecisionModelError):
        await client.decide("m", "sky", _CHOICE_QUESTION)


async def test_a_timeout_is_a_decision_error_too():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client = OpenRouterDecisionClient("test-key", transport=_transport(handler))
    with pytest.raises(DecisionModelError, match="ReadTimeout"):
        await client.decide("m", "sky", _CHOICE_QUESTION)


def test_without_the_gateway_key_there_is_no_decision_client(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(DecisionModelError, match="OPENROUTER_API_KEY"):
        intent_decisions.get_decision_client()


# ---------------------------------------------------------------- Choice rules


def test_family_option_keys_keep_display_names_unless_numeric_or_a_sentinel():
    assert intent_decisions.family_option_keys(["REFLEX HYBRID", "2024", "unclear", "REFLEX HYBRID"]) == {
        "REFLEX HYBRID": "REFLEX HYBRID",
        "family: 2024": "2024",
        "family: unclear": "unclear",
    }


def test_the_family_question_offers_every_family_plus_unclear_and_none():
    question = intent_decisions.family_question(
        intent_decisions.family_option_keys(_FAMILY_NAMES), follow_up=True
    )
    assert question["type"] == "choice"
    assert list(question["criteria"]) == [*_FAMILY_NAMES, "unclear", "no_family"]
    assert "`previous_family`" in question["instructions"]
    assert set(intent_decisions.question_type_question()["criteria"]) == set(QUESTION_TYPE_NAMES)


def test_each_family_option_carries_its_description():
    # Issue #30: a bare name was too thin to choose between families.
    question = intent_decisions.family_question(
        intent_decisions.family_option_keys(_FAMILY_NAMES),
        follow_up=False,
        family_descriptions={"REFLEX HYBRID": "Parts include template.", "REFLEX Foot Recon": ""},
    )

    assert question["criteria"]["REFLEX HYBRID"] == "The REFLEX HYBRID product family. Parts include template."
    # A family with no description (or an empty one) keeps the bare sentence.
    assert question["criteria"]["REFLEX Foot Recon"] == "The REFLEX Foot Recon product family."
    assert question["criteria"]["MIS - Foot Recon System"] == "The MIS - Foot Recon System product family."


_FAMILY_KEYS = intent_decisions.family_option_keys(_FAMILY_NAMES)


def test_a_family_at_or_above_the_threshold_is_picked():
    answer = ChoiceAnswer("REFLEX HYBRID", {"REFLEX HYBRID": 0.6, "REFLEX Foot Recon": 0.3})
    assert intent_decisions.pick_family(answer, _FAMILY_KEYS, 0.6, 4) == ("REFLEX HYBRID", [], True)


def test_a_family_below_the_threshold_asks_with_its_likely_candidates():
    answer = ChoiceAnswer(
        "REFLEX HYBRID",
        {
            "REFLEX HYBRID": 0.45,
            "REFLEX Foot Recon": 0.4,
            "MIS - Foot Recon System": 0.04,
            "unclear": 0.11,
        },
    )
    # Below the floor isn't offered; the sentinels never are.
    assert intent_decisions.pick_family(answer, _FAMILY_KEYS, 0.6, 4) == (
        None,
        ["REFLEX HYBRID", "REFLEX Foot Recon"],
        True,
    )
    assert intent_decisions.pick_family(answer, _FAMILY_KEYS, 0.6, 1)[1] == ["REFLEX HYBRID"]


def test_no_family_means_the_question_names_none_and_unclear_means_it_does():
    no_family = ChoiceAnswer("no_family", {"no_family": 0.9})
    unclear = ChoiceAnswer("unclear", {"unclear": 0.8, "REFLEX HYBRID": 0.1})
    assert intent_decisions.pick_family(no_family, _FAMILY_KEYS, 0.6, 4) == (None, [], False)
    assert intent_decisions.pick_family(unclear, _FAMILY_KEYS, 0.6, 4) == (None, ["REFLEX HYBRID"], True)


def test_a_runner_up_within_the_margin_is_detected_too():
    close = ChoiceAnswer("technique_procedural", {"technique_procedural": 0.5, "system_contents": 0.4})
    clear = ChoiceAnswer("technique_procedural", {"technique_procedural": 0.7, "system_contents": 0.2})
    assert intent_decisions.pick_question_types(close, 0.15) == ["technique_procedural", "system_contents"]
    assert intent_decisions.pick_question_types(clear, 0.15) == ["technique_procedural"]
    # A margin of 0 never asks.
    assert intent_decisions.pick_question_types(close, 0.0) == ["technique_procedural"]


def test_unclassified_is_never_paired():
    near_unclassified = ChoiceAnswer("system_contents", {"system_contents": 0.45, "unclassified": 0.4})
    unclassified_first = ChoiceAnswer("unclassified", {"unclassified": 0.45, "system_contents": 0.4})
    assert intent_decisions.pick_question_types(near_unclassified, 0.15) == ["system_contents"]
    assert intent_decisions.pick_question_types(unclassified_first, 0.15) == ["unclassified"]


# ---------------------------------------------------------------- the workflow's decision path


async def test_classify_intent_on_a_decision_model_asks_choices_and_never_calls_the_chat_model(monkeypatch):
    client = _FakeDecisionClient(
        _intent_answers(
            ("REFLEX HYBRID", {"REFLEX HYBRID": 0.92}),
            ("technique_procedural", {"technique_procedural": 0.5, "system_contents": 0.45}),
        )
    )
    monkeypatch.setattr(intent_decisions, "get_decision_client", lambda: client)
    monkeypatch.setattr(det, "get_chat_model", _no_chat_model)

    result = await det._classify_intent(
        "steps for the hybrid tray?",
        _FAMILY_NAMES,
        previous_family="MIS - Foot Recon System",
        previous_message="What screws are in the MIS tray?",
        slot_model=_DECISION_MODEL,
    )

    assert isinstance(result, det._IntentClassification)
    assert result.product_family == "REFLEX HYBRID"
    assert result.mentions_family is True
    assert result.question_type == ["technique_procedural", "system_contents"]
    [request] = client.requests
    assert request["model"] == "typesafe/jev-1.13-20260917"
    assert request["state"] == {
        "question": "steps for the hybrid tray?",
        "previous_question": "What screws are in the MIS tray?",
        "previous_family": "MIS - Foot Recon System",
    }
    assert set(request["questions"]) == {"product_family", "question_type"}


async def test_the_decision_path_sends_descriptions_and_reports_the_top_family_probabilities(monkeypatch):
    client = _FakeDecisionClient(
        _intent_answers(
            (
                "REFLEX Foot Recon",
                {
                    "REFLEX Foot Recon": 0.4,
                    "unclear": 0.2,
                    "REFLEX HYBRID": 0.15,
                    "MIS - Foot Recon System": 0.1,
                    "no_family": 0.1,
                    "family: 2024": 0.05,
                },
            )
        )
    )
    monkeypatch.setattr(intent_decisions, "get_decision_client", lambda: client)

    result = await det._classify_intent(
        "tetra staple sizes?",
        [*_FAMILY_NAMES, "2024"],
        family_descriptions={"REFLEX Foot Recon": "Parts include tetra."},
        slot_model=_DECISION_MODEL,
    )

    criteria = client.requests[0]["questions"]["product_family"]["criteria"]
    assert criteria["REFLEX Foot Recon"] == "The REFLEX Foot Recon product family. Parts include tetra."
    # Under the 0.6 default, so no pick -- but the eval still sees why.
    assert result.product_family is None
    assert result.family_probabilities == {
        "REFLEX Foot Recon": 0.4,
        "unclear": 0.2,
        "REFLEX HYBRID": 0.15,
        "MIS - Foot Recon System": 0.1,
        "no_family": 0.1,
    }


async def test_the_chat_path_lists_each_family_with_its_description(monkeypatch):
    prompts: list = []

    class _ChatModel:
        def with_structured_output(self, schema):
            class _Runnable:
                async def ainvoke(self, messages):
                    prompts.append(messages)
                    return schema.model_validate({"product_family": "REFLEX HYBRID"})

            return _Runnable()

    monkeypatch.setattr(det, "get_chat_model", lambda *arguments: _ChatModel())

    result = await det._classify_intent(
        "hybrid template?",
        _FAMILY_NAMES,
        family_descriptions={"REFLEX HYBRID": "Parts include template."},
        slot_model=runtime.SlotModel(Slot.INTENT_CLASSIFICATION),
    )

    system_prompt = prompts[0][0].content
    assert "- REFLEX HYBRID: Parts include template.\n" in system_prompt
    assert "- MIS - Foot Recon System\n" in system_prompt
    # The chat path reports no probabilities, and none leak into its schema.
    assert result.family_probabilities == {}
    assert "family_probabilities" not in det._IntentClassification.model_json_schema()["properties"]


async def test_with_no_families_only_the_question_type_is_asked(monkeypatch):
    client = _FakeDecisionClient({"question_type": ("unclassified", {"unclassified": 0.8})})
    monkeypatch.setattr(intent_decisions, "get_decision_client", lambda: client)

    result = await det._classify_intent("hello", [], slot_model=_DECISION_MODEL)

    assert (result.product_family, result.mentions_family, result.question_type) == (None, False, ["unclassified"])
    assert set(client.requests[0]["questions"]) == {"question_type"}


async def test_a_failed_decision_call_runs_the_default_chat_path_and_logs_it(monkeypatch, caplog):
    class _Failing:
        async def decide(self, model_id, state, questions):
            raise DecisionModelError("decisions request failed with 503: overloaded")

    calls: list = []

    class _ChatModel:
        def with_structured_output(self, schema):
            class _Runnable:
                async def ainvoke(self, messages):
                    return schema.model_validate({"product_family": None, "question_type": ["unclassified"]})

            return _Runnable()

    def fake_get_chat_model(*arguments):
        calls.append(arguments)
        return _ChatModel()

    monkeypatch.setattr(runtime.logger, "disabled", False)
    monkeypatch.setattr(intent_decisions, "get_decision_client", lambda: _Failing())
    monkeypatch.setattr(det, "get_chat_model", fake_get_chat_model)

    with caplog.at_level(logging.WARNING, logger=runtime.__name__):
        result = await det._classify_intent("What screws?", _FAMILY_NAMES, slot_model=_DECISION_MODEL)

    assert result.question_type == ["unclassified"]
    # The default: CHAT_MODEL, exactly the call an unset slot makes.
    assert calls == [()]
    [record] = [record for record in caplog.records if record.message.startswith("slot_output_fallback")]
    assert record.model_kind == "decision"
    assert "503" in record.reason


async def test_the_question_type_reply_is_a_choice_on_a_decision_model(monkeypatch):
    client = _FakeDecisionClient({"question_type": ("technique_procedural", {"technique_procedural": 0.9})})
    monkeypatch.setattr(intent_decisions, "get_decision_client", lambda: client)
    monkeypatch.setattr(det, "get_chat_model", _no_chat_model)
    monkeypatch.setattr(det, "slot_model_for_run", lambda slot: _DECISION_MODEL)

    assert await det._classify_question_type_reply("bunion steps?", "I need the steps") == "technique_procedural"
    assert client.requests[0]["state"] == {"question": "bunion steps?", "reply": "I need the steps"}


async def test_the_intent_eval_scores_the_decision_path_with_its_cost(monkeypatch):
    families = [
        ProductFamilyRecord(uuid.uuid4(), name, name) for name in _FAMILY_NAMES
    ]
    client = _FakeDecisionClient(
        _intent_answers(("REFLEX HYBRID", {"REFLEX HYBRID": 0.5, "REFLEX Foot Recon": 0.45})),
        cost=0.00001,
    )
    monkeypatch.setattr(intent_decisions, "get_decision_client", lambda: client)
    row = {
        "turns": [{"role": "user", "content": "the REFLEX tray?"}],
        "expected_product_family": None,
        "expected_question_type": ["product_characteristics"],
        "expects_family_clarification": True,
        "expects_question_type_clarification": False,
    }

    async def classify(*arguments, **keywords):
        return await det._classify_intent(*arguments, **keywords, slot_model=_DECISION_MODEL)

    result = await evaluate_row(row, families, classify)

    # 0.5 is under the default threshold, so the rep would be asked.
    assert result.would_clarify_family is True
    assert result.family_correct is True
    usage = result.token_usage["typesafe/jev-1.13-20260917"]
    assert (usage.input_tokens, usage.cost) == (120, 0.00001)


# ---------------------------------------------------------------- runtime


def test_a_decision_model_survives_the_run_config_round_trip():
    config = runtime.slot_models_config({Slot.INTENT_CLASSIFICATION.value: _DECISION_MODEL})
    choice = config[runtime.SLOT_MODELS_CONFIG_KEY]["intent_classification"]

    restored = runtime._slot_model_from_choice(Slot.INTENT_CLASSIFICATION, choice, choice["source"])

    assert restored == _DECISION_MODEL
    assert restored.is_decision_model
    assert restored.describe()["family_confidence_threshold"] == DEFAULT_FAMILY_CONFIDENCE_THRESHOLD
    # A chat model's description is unchanged by any of this.
    assert set(runtime.SlotModel(Slot.JUDGE, "vendor/x", "low", "setting").describe()) == {
        "model_id",
        "reasoning_effort",
        "source",
    }


# ---------------------------------------------------------------- probe


def _openrouter_client_answering(choice_for: dict[str, str]):
    """A real OpenRouterDecisionClient on a mocked transport that answers
    each question id with the given option (at 0.9).
    """

    def handler(request):
        questions = json.loads(request.content)["questions"]
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13-20260917",
                "answers": {
                    question_id: {
                        "type": "choice",
                        "choice": choice_for[question_id],
                        "probabilities": {choice_for[question_id]: 0.9},
                    }
                    for question_id in questions
                },
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )

    return OpenRouterDecisionClient("test-key", transport=httpx.MockTransport(handler))


async def test_the_decision_probe_passes_on_valid_keys(monkeypatch):
    client = _openrouter_client_answering(
        {"product_family": "MIS - Foot Recon System", "question_type": "product_characteristics"}
    )
    monkeypatch.setattr(probe, "build_probe_decision_client", lambda: client)

    result = await probe.probe_slot(Slot.INTENT_CLASSIFICATION, "typesafe/jev-1.13-20260917", None, "decision")

    assert result.passed, result.details
    assert len(result.details) == len(probe.DECISION_SLOT_CHECKS[Slot.INTENT_CLASSIFICATION])
    assert "family='MIS - Foot Recon System'" in result.details[0]


async def test_the_decision_probe_fails_when_the_family_isnt_one_of_its_keys(monkeypatch):
    client = _openrouter_client_answering(
        {"product_family": "Some Other Family", "question_type": "product_characteristics"}
    )
    monkeypatch.setattr(probe, "build_probe_decision_client", lambda: client)

    result = await probe.probe_slot(Slot.INTENT_CLASSIFICATION, "typesafe/jev-1.13-20260917", None, "decision")

    assert not result.passed
    assert result.details[0].startswith("FAIL ")
    assert "isn't one of its options" in result.details[0]


# ---------------------------------------------------------------- Settings routes

_CHAT_CATALOG = [
    {
        "id": "vendor/structured-model",
        "name": "Structured Model",
        "pricing": {"prompt": "0.000001", "completion": "0.000004"},
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": ["structured_outputs", "reasoning"],
    },
]
_DECISION_CATALOG = [
    {
        "id": "typesafe/jev-1.13",
        "canonical_slug": "typesafe/jev-1.13-20260917",
        "name": "TypeSafe: Jev 1.13",
        "pricing": {"prompt": "0.00000004", "completion": "0"},
        "architecture": {"input_modalities": ["text"], "output_modalities": ["decisions"]},
    },
    # An alias with no canonical slug isn't offered.
    {
        "id": "~typesafe/jev-latest",
        "name": "TypeSafe: Jev (latest)",
        "architecture": {"output_modalities": ["decisions"]},
    },
    # Nor is one whose canonical slug is the alias itself, as OpenRouter
    # lists `~typesafe/jev-latest` today.
    {
        "id": "~typesafe/jev-latest-too",
        "canonical_slug": "~typesafe/jev-latest-too",
        "name": "TypeSafe: Jev (latest, slugged)",
        "architecture": {"output_modalities": ["decisions"]},
    },
]


def _run_isolated(coroutine):
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


@pytest.fixture
def decision_gateway(monkeypatch):
    """OPENROUTER_API_KEY set, both catalogs faked, the decision probe on a
    client answering valid keys, and an empty table before and after.
    """

    async def clear() -> None:
        for slot in Slot:
            await repository.delete_assignment(slot.value)

    async def fake_chat_catalog():
        return _CHAT_CATALOG

    async def fake_decision_catalog():
        return _DECISION_CATALOG

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(catalog, "fetch_openrouter_models", fake_chat_catalog)
    monkeypatch.setattr(catalog, "fetch_openrouter_decision_models", fake_decision_catalog)
    monkeypatch.setattr(
        probe,
        "build_probe_decision_client",
        lambda: _openrouter_client_answering(
            {"product_family": "REFLEX HYBRID", "question_type": "technique_procedural"}
        ),
    )
    catalog.clear_catalog_cache()
    _run_isolated(clear())
    yield
    _run_isolated(clear())
    catalog.clear_catalog_cache()


def _admin_headers(client: TestClient, monkeypatch) -> dict:
    email = f"test-{uuid.uuid4().hex[:12]}@example.com"
    monkeypatch.setenv("ADMIN_EMAILS", email)
    response = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_only_intent_classification_lists_decision_models_by_dated_id(decision_gateway, monkeypatch):
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        intent_models = client.get("/admin/model-assignments/intent_classification/models", headers=headers)
        judge_models = client.get("/admin/model-assignments/judge/models", headers=headers)
        slots = client.get("/admin/model-assignments", headers=headers).json()["slots"]

    listed = [(model["model_id"], model["model_kind"]) for model in intent_models.json()]
    assert listed == [
        ("vendor/structured-model", "chat"),
        ("typesafe/jev-1.13-20260917", "decision"),
    ]
    assert intent_models.json()[1]["supports_reasoning"] is False
    assert [model["model_kind"] for model in judge_models.json()] == ["chat"]
    accepts = {slot["slot"]: slot["accepts_decision_models"] for slot in slots}
    assert accepts == {slot.value: slot is Slot.INTENT_CLASSIFICATION for slot in Slot}


def test_saving_a_decision_model_pins_its_dated_id_and_stores_its_thresholds(decision_gateway, monkeypatch):
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        response = client.put(
            "/admin/model-assignments/intent_classification",
            headers=headers,
            json={"model_id": "typesafe/jev-1.13", "model_kind": "decision", "question_type_margin": 0.1},
        )

    assert response.status_code == 200, response.text
    intent = next(slot for slot in response.json()["slots"] if slot["slot"] == "intent_classification")
    assert intent["assignment"]["model_id"] == "typesafe/jev-1.13-20260917"
    assert intent["assignment"]["model_kind"] == "decision"
    assert intent["assignment"]["reasoning_effort"] is None
    assert intent["assignment"]["family_confidence_threshold"] == DEFAULT_FAMILY_CONFIDENCE_THRESHOLD
    assert intent["assignment"]["question_type_margin"] == 0.1

    repository.invalidate_cache()
    saved = _run_isolated(runtime.configured_slot_model(Slot.INTENT_CLASSIFICATION))
    assert saved.is_decision_model
    assert (saved.family_confidence_threshold, saved.question_type_margin) == (
        DEFAULT_FAMILY_CONFIDENCE_THRESHOLD,
        0.1,
    )


def test_a_decision_model_sent_as_a_chat_model_is_saved_as_a_decision_model(decision_gateway, monkeypatch):
    # Production, 2026-10-18: picking Jev in Settings reached the server as a
    # chat model and was refused as "isn't in OpenRouter's model catalog".
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        response = client.put(
            "/admin/model-assignments/intent_classification",
            headers=headers,
            json={"model_id": "typesafe/jev-1.13-20260917", "reasoning_effort": None},
        )
        with_effort = client.put(
            "/admin/model-assignments/intent_classification",
            headers=headers,
            json={"model_id": "typesafe/jev-1.13-20260917", "reasoning_effort": "low"},
        )
        on_judge = client.put(
            "/admin/model-assignments/judge",
            headers=headers,
            json={"model_id": "typesafe/jev-1.13-20260917"},
        )

    assert response.status_code == 200, response.text
    intent = next(slot for slot in response.json()["slots"] if slot["slot"] == "intent_classification")
    assert intent["assignment"]["model_kind"] == "decision"
    assert intent["assignment"]["model_id"] == "typesafe/jev-1.13-20260917"
    assert with_effort.status_code == 400
    assert "takes no reasoning effort" in with_effort.json()["detail"]
    # A slot that takes no decision model still finds no such chat model.
    assert on_judge.status_code == 400
    assert "isn't in OpenRouter's model catalog" in on_judge.json()["detail"]


def test_a_decision_model_failing_its_probe_isnt_saved(decision_gateway, monkeypatch):
    monkeypatch.setattr(
        probe,
        "build_probe_decision_client",
        lambda: _openrouter_client_answering({"product_family": "Nope", "question_type": "unclassified"}),
    )
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        response = client.put(
            "/admin/model-assignments/intent_classification",
            headers=headers,
            json={"model_id": "typesafe/jev-1.13-20260917", "model_kind": "decision"},
        )
        listed = client.get("/admin/model-assignments", headers=headers).json()

    assert response.status_code == 422
    assert "failed the Intent classification probe" in response.json()["detail"]
    assert all(slot["assignment"] is None for slot in listed["slots"])


@pytest.mark.parametrize(
    ("slot_name", "body", "expected_status"),
    [
        # Only intent classification takes a decision model.
        ("judge", {"model_id": "typesafe/jev-1.13-20260917", "model_kind": "decision"}, 400),
        # Not in the decision listing (a chat model, say).
        ("intent_classification", {"model_id": "vendor/structured-model", "model_kind": "decision"}, 400),
        # A decision model takes no effort, and a chat model no thresholds.
        (
            "intent_classification",
            {"model_id": "typesafe/jev-1.13-20260917", "model_kind": "decision", "reasoning_effort": "low"},
            422,
        ),
        ("intent_classification", {"model_id": "vendor/structured-model", "question_type_margin": 0.1}, 422),
        (
            "intent_classification",
            {"model_id": "typesafe/jev-1.13-20260917", "model_kind": "decision", "family_confidence_threshold": 0},
            422,
        ),
    ],
)
def test_a_decision_model_is_refused_before_the_probe_when_it_cant_run(
    decision_gateway, monkeypatch, slot_name, body, expected_status
):
    with TestClient(app) as client:
        headers = _admin_headers(client, monkeypatch)
        response = client.put(f"/admin/model-assignments/{slot_name}", headers=headers, json=body)
    assert response.status_code == expected_status


def test_a_decision_override_is_refused_on_a_slot_that_takes_none(decision_gateway, monkeypatch):
    with TestClient(app) as client:
        response = client.post(
            "/chat/deterministic/stream",
            headers=_admin_headers(client, monkeypatch),
            json={
                "message": "What screws?",
                "model_overrides": {
                    "deterministic_answer": {"model_id": "typesafe/jev-1.13-20260917", "model_kind": "decision"}
                },
            },
        )
    assert response.status_code == 400
    assert "deterministic_answer" in response.json()["detail"]


def test_a_decision_override_reaches_the_turn_as_a_decision_model(decision_gateway):
    turn_models = _run_isolated(
        runtime.resolve_turn_slot_models(
            {
                "intent_classification": {
                    "model_id": "typesafe/jev-1.13-20260917",
                    "reasoning_effort": None,
                    "model_kind": "decision",
                    "family_confidence_threshold": 0.7,
                    "question_type_margin": None,
                }
            }
        )
    )
    intent = turn_models["intent_classification"]
    assert intent.is_decision_model
    assert (intent.family_confidence_threshold, intent.question_type_margin) == (0.7, DEFAULT_QUESTION_TYPE_MARGIN)


# ---------------------------------------------------------------- harness


async def test_the_harness_runs_a_listed_decision_model_as_one(decision_gateway):
    slot_models = await harness.resolve_eval_slot_models(
        [(Slot.INTENT_CLASSIFICATION, "typesafe/jev-1.13", None)], family_confidence_threshold=0.5
    )

    intent = slot_models[Slot.INTENT_CLASSIFICATION]
    assert intent.is_decision_model
    assert intent.model_id == "typesafe/jev-1.13-20260917"
    assert (intent.family_confidence_threshold, intent.question_type_margin) == (0.5, DEFAULT_QUESTION_TYPE_MARGIN)
    assert (
        "intent_classification: typesafe/jev-1.13-20260917 (decision model: family confidence "
        "threshold 0.5, question-type margin 0.15; override)"
    ) in harness.format_slot_models(slot_models)
    # A chat id is still a chat model.
    chat = await harness.resolve_eval_slot_models([(Slot.INTENT_CLASSIFICATION, "vendor/structured-model", "low")])
    assert not chat[Slot.INTENT_CLASSIFICATION].is_decision_model


async def test_the_harness_refuses_thresholds_without_a_decision_model_and_effort_with_one(decision_gateway):
    with pytest.raises(harness.EvalModelError):
        await harness.resolve_eval_slot_models([], question_type_margin=0.1)
    with pytest.raises(harness.EvalModelError):
        await harness.resolve_eval_slot_models([(Slot.INTENT_CLASSIFICATION, "typesafe/jev-1.13", "low")])

"""config/llm_clients.py: which chat client get_chat_model() builds for a
given environment, CHAT_MODEL's startup validation, and the OpenAI-direct
fallback on gateway failures. No network and no real keys: building a
ChatOpenAI makes no request, and the fallback runs against fake chat
models.
"""
import json
import logging

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import llm_clients
from config.llm_clients import (
    OPENROUTER_BASE_URL,
    ChatModelConfigError,
    configured_chat_model,
    get_chat_model,
    with_openai_direct_fallback,
)


@pytest.fixture(autouse=True)
def chat_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    for name in ("OPENROUTER_API_KEY", "CHAT_MODEL", "OPENAI_CHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    get_chat_model.cache_clear()
    yield monkeypatch
    get_chat_model.cache_clear()


def _secret(model: ChatOpenAI) -> str:
    return model.openai_api_key.get_secret_value()


def test_without_gateway_key_calls_openai_direct_with_prefix_stripped(chat_environment):
    chat_environment.setenv("CHAT_MODEL", "openai/gpt-5.6-luna")

    model = get_chat_model()

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-5.6-luna"
    assert model.openai_api_base is None
    assert _secret(model) == "sk-openai-test"
    assert model.use_responses_api is True
    # Today's OpenAI-direct behavior: store left to the server's default.
    assert model.store is None


def test_unset_chat_model_defaults_to_gpt_4o():
    assert configured_chat_model() == "openai/gpt-4o"
    assert get_chat_model().model_name == "gpt-4o"


def test_gateway_key_routes_through_openrouter_with_openai_direct_fallback(chat_environment):
    chat_environment.setenv("CHAT_MODEL", "openai/gpt-5.6-luna")
    chat_environment.setenv("OPENROUTER_API_KEY", "sk-or-test")

    model = get_chat_model()

    assert isinstance(model, RunnableWithFallbacks)
    gateway_model = model.runnable
    assert gateway_model.model_name == "openai/gpt-5.6-luna"
    assert gateway_model.openai_api_base == OPENROUTER_BASE_URL
    assert _secret(gateway_model) == "sk-or-test"
    assert gateway_model.use_responses_api is True
    assert gateway_model.store is False
    [fallback_model] = model.fallbacks
    assert fallback_model.model_name == "gpt-5.6-luna"
    assert fallback_model.openai_api_base is None
    assert _secret(fallback_model) == "sk-openai-test"
    assert fallback_model.store is False


def test_fallback_runs_chat_model_even_when_the_call_asked_for_another(chat_environment):
    chat_environment.setenv("CHAT_MODEL", "openai/gpt-5.6-luna")
    chat_environment.setenv("OPENROUTER_API_KEY", "sk-or-test")

    model = get_chat_model("anthropic/claude-sonnet-5")

    assert model.runnable.model_name == "anthropic/claude-sonnet-5"
    assert model.fallbacks[0].model_name == "gpt-5.6-luna"


def test_gateway_requests_never_replay_server_stored_item_ids(chat_environment):
    # OpenRouter's Responses API is stateless, so react_agent's second
    # round can't reference the first round's reasoning/message item ids.
    chat_environment.setenv("OPENROUTER_API_KEY", "sk-or-test")
    earlier_response = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_earlier", "summary": []},
            {"type": "text", "text": "Looking that up.", "id": "msg_earlier"},
        ]
    )

    payload = get_chat_model().runnable._get_request_payload(
        [HumanMessage(content="which screw?"), earlier_response, HumanMessage(content="and the wire?")]
    )

    assert payload["store"] is False
    assert "previous_response_id" not in payload
    replayed = json.dumps(payload["input"])
    assert "rs_earlier" not in replayed
    assert "msg_earlier" not in replayed
    assert "Looking that up." in replayed


def test_non_openai_model_without_gateway_key_is_rejected():
    with pytest.raises(ChatModelConfigError, match="OPENROUTER_API_KEY"):
        get_chat_model("anthropic/claude-sonnet-5")


def test_reasoning_effort_reaches_both_sides(chat_environment):
    chat_environment.setenv("OPENROUTER_API_KEY", "sk-or-test")

    model = get_chat_model(reasoning_effort="low")

    assert model.runnable.reasoning_effort == "low"
    assert model.fallbacks[0].reasoning_effort == "low"
    assert get_chat_model().runnable.reasoning_effort is None


def test_cache_is_keyed_by_model_and_effort():
    assert get_chat_model() is get_chat_model()
    assert get_chat_model(reasoning_effort="low") is get_chat_model(reasoning_effort="low")
    assert get_chat_model(reasoning_effort="low") is not get_chat_model(reasoning_effort="high")
    assert get_chat_model("openai/gpt-4o-mini") is not get_chat_model()


@pytest.mark.parametrize(
    ("chat_model", "gateway_key"),
    [
        ("gpt-5.6-luna", None),
        ("anthropic/claude-sonnet-5", "sk-or-test"),
        ("anthropic/claude-sonnet-5", None),
        ("openai/", "sk-or-test"),
    ],
)
def test_chat_model_that_isnt_an_openai_router_id_fails_validation(
    chat_environment, chat_model, gateway_key
):
    chat_environment.setenv("CHAT_MODEL", chat_model)
    if gateway_key:
        chat_environment.setenv("OPENROUTER_API_KEY", gateway_key)

    with pytest.raises(ChatModelConfigError, match="CHAT_MODEL must be"):
        configured_chat_model()


def test_leftover_openai_chat_model_fails_instead_of_falling_back_to_the_default(
    chat_environment,
):
    chat_environment.setenv("OPENAI_CHAT_MODEL", "gpt-5.6-luna")

    with pytest.raises(ChatModelConfigError, match="CHAT_MODEL=openai/gpt-5.6-luna"):
        configured_chat_model()


def test_app_startup_fails_on_an_invalid_chat_model(chat_environment):
    from api.main import app

    chat_environment.setenv("CHAT_MODEL", "gpt-5.6-luna")

    with pytest.raises(ChatModelConfigError), TestClient(app):
        pass


class _FailingChatModel(BaseChatModel):
    """Stands in for the OpenRouter side: every call raises `error`."""

    error: Exception
    model_name: str = "openai/gpt-5.6-luna"

    @property
    def _llm_type(self) -> str:
        return "failing-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise self.error


class _NamedFakeChatModel(FakeListChatModel):
    model_name: str = "gpt-5.6-luna"


def _status_error(error_class: type[openai.APIStatusError], status_code: int) -> Exception:
    request = httpx.Request("POST", f"{OPENROUTER_BASE_URL}/responses")
    return error_class(
        "gateway said no", response=httpx.Response(status_code, request=request), body=None
    )


def _gateway_failures() -> list[Exception]:
    request = httpx.Request("POST", f"{OPENROUTER_BASE_URL}/responses")
    return [
        _status_error(openai.InternalServerError, 502),
        openai.APITimeoutError(request=request),
        openai.APIConnectionError(request=request),
        _status_error(openai.AuthenticationError, 401),
        # Out of OpenRouter credit (see the 402 mapping test below).
        _status_error(llm_clients.OpenRouterOutOfCredits, 402),
        # Rate limit or quota reached on OpenRouter.
        _status_error(openai.RateLimitError, 429),
    ]


@pytest.fixture
def caplog(caplog, monkeypatch):
    # Alembic's env.py runs logging.config.fileConfig, which disables every
    # logger that already exists, this one included, once an earlier test
    # has run migrations.
    monkeypatch.setattr(llm_clients.logger, "disabled", False)
    return caplog


@pytest.mark.parametrize("gateway_error", _gateway_failures(), ids=lambda error: type(error).__name__)
async def test_gateway_failure_falls_back_to_openai_direct_and_logs(gateway_error, caplog):
    model = with_openai_direct_fallback(
        _FailingChatModel(error=gateway_error), _NamedFakeChatModel(responses=["from openai direct"])
    )

    with caplog.at_level(logging.WARNING, logger=llm_clients.__name__):
        response = await model.ainvoke("which screw?")

    assert response.text == "from openai direct"
    [record] = [
        record for record in caplog.records if getattr(record, "event", None) == "chat_gateway_fallback"
    ]
    assert record.error_type == type(gateway_error).__name__
    assert record.gateway_model == "openai/gpt-5.6-luna"
    assert record.fallback_model == "gpt-5.6-luna"
    assert "chat_gateway_fallback" in record.getMessage()


@pytest.mark.parametrize(
    "request_error",
    [
        _status_error(openai.BadRequestError, 400),
        _status_error(openai.PermissionDeniedError, 403),
        OutputParserException("not the schema"),
    ],
    ids=lambda error: type(error).__name__,
)
async def test_request_or_output_errors_do_not_fall_back(request_error, caplog):
    direct_model = _NamedFakeChatModel(responses=["from openai direct"])
    model = with_openai_direct_fallback(_FailingChatModel(error=request_error), direct_model)

    with (
        caplog.at_level(logging.WARNING, logger=llm_clients.__name__),
        pytest.raises(type(request_error)),
    ):
        await model.ainvoke("which screw?")

    assert direct_model.i == 0  # the fake's own call counter: never reached
    assert not [record for record in caplog.records if getattr(record, "event", None)]


def _part_lookup(sku: str) -> str:
    """Looks up a part by SKU."""
    return sku


async def test_tool_bound_call_still_falls_back():
    # The `-> Runnable` annotations matter: with_fallbacks only re-wraps a
    # method annotated as returning one, and otherwise hands back the
    # gateway side alone.
    class _ToolCallingFake(_NamedFakeChatModel):
        def bind_tools(self, tools, **kwargs) -> Runnable:
            return self.bind(tools=[tool.__name__ for tool in tools])

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=f"tools={kwargs['tools']}"))]
            )

    class _FailingToolCallingFake(_FailingChatModel):
        def bind_tools(self, tools, **kwargs) -> Runnable:
            return self.bind(tools=[tool.__name__ for tool in tools])

    model = with_openai_direct_fallback(
        _FailingToolCallingFake(error=_status_error(openai.InternalServerError, 503)),
        _ToolCallingFake(responses=["unused"]),
    )

    response = await model.bind_tools([_part_lookup]).ainvoke("which screw?")

    assert response.text == "tools=['_part_lookup']"


def test_real_chat_model_keeps_the_fallback_through_tools_and_structured_output(
    chat_environment,
):
    chat_environment.setenv("OPENROUTER_API_KEY", "sk-or-test")

    class _Answer(BaseModel):
        sku: str

    tool_bound = get_chat_model().bind_tools([_part_lookup])
    structured = get_chat_model().with_structured_output(_Answer)

    assert isinstance(tool_bound, RunnableWithFallbacks)
    for binding in (tool_bound.runnable, *tool_bound.fallbacks):
        assert [tool["name"] for tool in _tool_definitions(binding)] == ["_part_lookup"]
    assert isinstance(structured, RunnableWithFallbacks)
    assert len(structured.fallbacks) == 1


def _tool_definitions(binding) -> list[dict]:
    return [tool.get("function", tool) for tool in binding.kwargs["tools"]]


def test_the_gateway_client_raises_a_402_as_out_of_credits(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    gateway_model = llm_clients.openrouter_chat_model("openai/gpt-5.6-luna")
    request = httpx.Request("POST", f"{OPENROUTER_BASE_URL}/responses")

    for client in (gateway_model.root_client, gateway_model.root_async_client):
        out_of_credits = client._make_status_error("no credit", body=None, response=httpx.Response(402, request=request))
        forbidden = client._make_status_error("no", body=None, response=httpx.Response(403, request=request))
        assert isinstance(out_of_credits, llm_clients.OpenRouterOutOfCredits)
        assert isinstance(forbidden, openai.PermissionDeniedError)

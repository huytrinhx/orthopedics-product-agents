"""Chat and embedding client factories.

Every chat call goes through OpenRouter's OpenAI-compatible API once
`OPENROUTER_API_KEY` is set, so trying a different model is a `CHAT_MODEL`
change rather than new client code; without that key (local dev, CI) it
goes to OpenAI direct, exactly as before the gateway existed. `ChatOpenAI`
is the client either way, only its `base_url` and key change. A gateway
outage falls back to OpenAI direct for the call that hit it. Embeddings
always stay on OpenAI direct: the `text-embedding-3-small` vectors already
stored in pgvector would all need re-embedding otherwise. See
docs/adr/0007-openrouter-chat-gateway.md.
"""
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import httpx
import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableWithFallbacks
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import tiktoken_cache  # noqa: F401  (sets TIKTOKEN_CACHE_DIR)

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Written as an OpenRouter model id even when no gateway is configured, so
# the same value works with and without OPENROUTER_API_KEY.
DEFAULT_CHAT_MODEL = "openai/gpt-4o"
OPENAI_MODEL_PREFIX = "openai/"

# Responses API, not Chat Completions, for every node: GPT-5.x reasoning
# models reject function tools alongside reasoning on
# /v1/chat/completions, and react_agent's tool-bound `generate` call is
# exactly where the reasoning matters (tool choice and when to stop).
# OpenRouter's Chat Completions handled every node, reasoning and tools
# included (backend/evals/openrouter_spike.py, 2026-10-17, ADR 0007), but
# the OpenAI-direct fallback below shares this setting and still rejects
# that combination there -- so Responses stays the single path. Responses-API
# messages carry content as a list of blocks, so read model output with
# `.text`, not `.content`.
USE_RESPONSES_API = True

class OpenRouterOutOfCredits(openai.APIStatusError):
    """The 402 OpenRouter returns when the account is out of credit. The
    openai SDK has no class of its own for a 402 (it raises a plain
    APIStatusError), and with_fallbacks matches with an `except` clause, so
    openrouter_chat_model's clients raise this instead (_raise_out_of_credits_distinctly).
    """


def _raise_out_of_credits_distinctly(client: Any) -> None:
    make_status_error = client._make_status_error

    def make_status_error_with_402(err_msg: str, *, body: object, response: Any) -> openai.APIStatusError:
        if response.status_code == 402:
            return OpenRouterOutOfCredits(err_msg, response=response, body=body)
        return make_status_error(err_msg, body=body, response=response)

    client._make_status_error = make_status_error_with_402


# The failures that mean the gateway itself is down, misconfigured, or out
# of capacity for us: a 5xx, a timeout or dropped connection
# (APITimeoutError is an APIConnectionError), a rejected key, a 402 (out of
# credit), or a 429 (rate limit or quota reached -- the OpenAI client has
# already retried it twice by then). OpenAI direct has its own, separate
# limits, so it can still answer. Any other 4xx about the request (400,
# 403), or a model output that doesn't parse, would fail the same way on
# OpenAI direct, so those still raise.
GATEWAY_FAILURES: tuple[type[Exception], ...] = (
    openai.InternalServerError,
    openai.APIConnectionError,
    openai.AuthenticationError,
    openai.RateLimitError,
    OpenRouterOutOfCredits,
)


class ChatModelConfigError(RuntimeError):
    pass


def configured_chat_model() -> str:
    """The validated `CHAT_MODEL`, an OpenRouter model id. Called at
    startup (api/main.py's lifespan) so a bad value stops the deploy
    instead of failing the first chat turn.

    It has to be an `openai/...` model, gateway or not: the OpenAI-direct
    path, which is also the gateway-outage fallback, runs it with the
    prefix stripped, so anything else would leave the fallback with no
    model it can call.
    """
    chat_model = os.environ.get("CHAT_MODEL")
    if chat_model is None:
        legacy_model = os.environ.get("OPENAI_CHAT_MODEL")
        if legacy_model:
            # Failing beats quietly dropping a deploy that still sets the
            # old variable back to DEFAULT_CHAT_MODEL.
            raise ChatModelConfigError(
                "OPENAI_CHAT_MODEL was replaced by CHAT_MODEL, which takes an OpenRouter "
                f"model id: set CHAT_MODEL={OPENAI_MODEL_PREFIX}{legacy_model} and remove "
                "OPENAI_CHAT_MODEL."
            )
        return DEFAULT_CHAT_MODEL
    if not chat_model.startswith(OPENAI_MODEL_PREFIX) or chat_model == OPENAI_MODEL_PREFIX:
        raise ChatModelConfigError(
            f"CHAT_MODEL must be an OpenRouter id for an OpenAI model "
            f"({OPENAI_MODEL_PREFIX}<model>, e.g. openai/gpt-5.6-luna), got {chat_model!r}. "
            "OpenAI direct runs it when OPENROUTER_API_KEY is unset, and as the fallback "
            "when the gateway is down."
        )
    return chat_model


def _openai_direct_model_name(model_id: str) -> str:
    if not model_id.startswith(OPENAI_MODEL_PREFIX):
        raise ChatModelConfigError(
            f"{model_id!r} isn't an OpenAI model, so it can only run through OpenRouter "
            "(set OPENROUTER_API_KEY)."
        )
    return model_id.removeprefix(OPENAI_MODEL_PREFIX)


def openai_direct_chat_model(
    model_id: str, *, reasoning_effort: str | None = None, store: bool | None = None
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        model=_openai_direct_model_name(model_id),
        use_responses_api=USE_RESPONSES_API,
        reasoning_effort=reasoning_effort,
        store=store,
    )


def openrouter_chat_model(
    model_id: str,
    *,
    reasoning_effort: str | None = None,
    use_responses_api: bool = USE_RESPONSES_API,
) -> ChatOpenAI:
    """`use_responses_api` is only overridden by the spike script, which
    runs every call shape under both APIs.
    """
    gateway_model = ChatOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        model=model_id,
        use_responses_api=use_responses_api,
        reasoning_effort=reasoning_effort,
        # OpenRouter's Responses API is stateless: it rejects `store: true`
        # and `previous_response_id`. ChatOpenAI sends neither by default,
        # but it only stops replaying earlier responses' item ids (which a
        # stateless server can't resolve) on react_agent's tool rounds when
        # store is explicitly False.
        store=False if use_responses_api else None,
    )
    for client in (gateway_model.root_client, gateway_model.root_async_client):
        if client is not None:
            _raise_out_of_credits_distinctly(client)
    return gateway_model


class _GatewayFailureLogger(BaseCallbackHandler):
    """Logs one line whenever a gateway call fails in a way that sends it to
    the OpenAI-direct fallback. `with_fallbacks` retries silently, and an
    outage that only shows up as slower answers would otherwise go unseen.
    """

    run_inline = True

    def __init__(self, gateway_model: str, fallback_model: str) -> None:
        self.gateway_model = gateway_model
        self.fallback_model = fallback_model

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        if not isinstance(error, GATEWAY_FAILURES):
            return
        status_code = getattr(error, "status_code", None)
        logger.warning(
            "chat_gateway_fallback gateway=openrouter model=%s fallback_model=%s "
            "error_type=%s status_code=%s",
            self.gateway_model,
            self.fallback_model,
            type(error).__name__,
            status_code,
            extra={
                "event": "chat_gateway_fallback",
                "gateway_model": self.gateway_model,
                "fallback_model": self.fallback_model,
                "error_type": type(error).__name__,
                "status_code": status_code,
            },
        )


def with_openai_direct_fallback(
    gateway_model: BaseChatModel, direct_model: BaseChatModel
) -> RunnableWithFallbacks:
    """Retries a call that hit a gateway failure (GATEWAY_FAILURES) against
    `direct_model`. `bind_tools`/`with_structured_output` on the result
    apply to both sides, so callers use it like a plain chat model.
    """
    failure_logger = _GatewayFailureLogger(
        gateway_model=getattr(gateway_model, "model_name", type(gateway_model).__name__),
        fallback_model=getattr(direct_model, "model_name", type(direct_model).__name__),
    )
    logged_gateway_model = gateway_model.model_copy(update={"callbacks": [failure_logger]})
    return logged_gateway_model.with_fallbacks([direct_model], exceptions_to_handle=GATEWAY_FAILURES)


@lru_cache
def get_chat_model(
    model: str | None = None, reasoning_effort: str | None = None
) -> ChatOpenAI | RunnableWithFallbacks:
    """`model` is an OpenRouter model id and defaults to CHAT_MODEL;
    `reasoning_effort` defaults to the model's own. Cached per (model,
    effort) pair.
    """
    model_id = model or configured_chat_model()
    if not os.environ.get("OPENROUTER_API_KEY"):
        return openai_direct_chat_model(model_id, reasoning_effort=reasoning_effort)
    # The fallback runs CHAT_MODEL even when this call asked for another
    # model, which may not exist on OpenAI direct. It's stateless like the
    # gateway, because a react_agent turn that falls back mid-loop replays
    # history the gateway produced, whose item ids OpenAI never stored.
    return with_openai_direct_fallback(
        openrouter_chat_model(model_id, reasoning_effort=reasoning_effort),
        openai_direct_chat_model(
            configured_chat_model(), reasoning_effort=reasoning_effort, store=False
        ),
    )


# ---------------------------------------------------------------- decision models
#
# A decision model (issue #23) writes no text: given a state and typed
# questions it answers each in one forward pass, a Choice with a probability
# for every option. OpenRouter carries them on its Decisions API, a separate
# endpoint from chat, with the same key -- so, like chat, there's no second
# vendor account, and trying Jev or Microsoft Decision-1 is a model id
# change. Only the intent-classification slot offers them
# (model_assignments/slots.py); agents/intent_decisions.py turns its
# questions into Choices. See docs/adr/0008-decision-models-for-intent.md.
#
# There's no OpenAI-direct counterpart, so no gateway-outage fallback here:
# a decision call that fails for any reason raises DecisionModelError, and
# the caller runs the slot's default chat path instead.

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
# A decision is one forward pass (about 255 ms over the WAN in one
# independent run); anything near this means the call is better answered
# by the chat fallback.
DECISION_TIMEOUT_SECONDS = 15.0


class DecisionModelError(RuntimeError):
    """Any decision call whose answer can't be used: the request failed
    (HTTP error, timeout, no gateway key) or the response didn't answer
    every question with one of its own options.
    """


@dataclass(frozen=True)
class ChoiceAnswer:
    # One of the question's own option keys.
    choice: str
    # Option key -> probability, covering every option the model scored.
    probabilities: dict[str, float]


@dataclass(frozen=True)
class DecisionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    # US dollars, as the provider reports it; 0 when it reports none.
    cost: float = 0.0


@dataclass(frozen=True)
class DecisionResponse:
    # The exact build that answered, which can be more specific than the
    # requested id.
    model: str
    answers: dict[str, ChoiceAnswer]
    usage: DecisionUsage


class DecisionClient(Protocol):
    """The one interface a decision provider implements. Only Choice
    questions are asked today: `questions` maps a question id to
    `{"type": "choice", "instructions": ..., "criteria": {option: description}}`.
    """

    async def decide(self, model_id: str, state: dict, questions: dict) -> DecisionResponse: ...


# Every decision response's usage within a `collect_decision_usage()`
# block, keyed by the model that answered -- the intent eval's tokens and
# cost per turn. LangChain's usage callback never sees these calls.
_decision_usage: ContextVar[dict[str, list[DecisionUsage]] | None] = ContextVar(
    "decision_usage", default=None
)


@contextmanager
def collect_decision_usage() -> Iterator[dict[str, list[DecisionUsage]]]:
    collected: dict[str, list[DecisionUsage]] = {}
    token = _decision_usage.set(collected)
    try:
        yield collected
    finally:
        _decision_usage.reset(token)


def record_decision_usage(response: DecisionResponse) -> None:
    """Called with every response a node acts on, whichever provider
    answered it; a no-op outside `collect_decision_usage()`.
    """
    collected = _decision_usage.get()
    if collected is not None:
        collected.setdefault(response.model, []).append(response.usage)


def _parse_choice_answer(question_id: str, question: dict, raw_answer: object) -> ChoiceAnswer:
    options = set(question["criteria"])
    if not isinstance(raw_answer, dict) or raw_answer.get("type") != "choice":
        raise DecisionModelError(f"no choice answer for {question_id!r}: {str(raw_answer)[:200]}")
    choice = raw_answer.get("choice")
    if choice not in options:
        raise DecisionModelError(f"{question_id!r} answered {choice!r}, which isn't one of its options")
    raw_probabilities = raw_answer.get("probabilities")
    if isinstance(raw_probabilities, dict):
        probabilities = {
            option: float(probability)
            for option, probability in raw_probabilities.items()
            if option in options and isinstance(probability, int | float)
        }
    elif isinstance(raw_answer.get("confidence"), int | float):
        # Probabilities are optional in the schema; the confidence is the
        # chosen option's, which is all a threshold needs.
        probabilities = {choice: float(raw_answer["confidence"])}
    else:
        raise DecisionModelError(f"{question_id!r} came back with no probabilities")
    return ChoiceAnswer(choice=choice, probabilities=probabilities)


class OpenRouterDecisionClient:
    """OpenRouter's Decisions API: `POST /api/alpha/decisions` with the
    gateway's key. `transport` is for tests (an httpx.MockTransport).
    """

    def __init__(
        self,
        api_key: str,
        url: str = OPENROUTER_DECISIONS_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.transport = transport

    async def decide(self, model_id: str, state: dict, questions: dict) -> DecisionResponse:
        # A client per call: the eval harness and the tests each run their
        # own event loops, which a shared connection pool can't span.
        try:
            async with httpx.AsyncClient(
                timeout=DECISION_TIMEOUT_SECONDS, transport=self.transport
            ) as client:
                response = await client.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": model_id, "state": state, "questions": questions},
                )
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as error:
            raise DecisionModelError(
                f"decisions request failed with {error.response.status_code}: "
                f"{error.response.text[:200]}"
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise DecisionModelError(f"decisions request failed: {type(error).__name__}: {error}") from error

        raw_answers = body.get("answers") if isinstance(body, dict) else None
        if not isinstance(raw_answers, dict):
            raise DecisionModelError(f"decisions response has no answers: {str(body)[:200]}")
        answers = {
            question_id: _parse_choice_answer(question_id, question, raw_answers.get(question_id))
            for question_id, question in questions.items()
        }
        raw_usage = body.get("usage") or {}
        usage = DecisionUsage(
            input_tokens=int(raw_usage.get("input_tokens") or 0),
            output_tokens=int(raw_usage.get("output_tokens") or 0),
            cost=float(raw_usage.get("cost") or 0.0),
        )
        return DecisionResponse(model=str(body.get("model") or model_id), answers=answers, usage=usage)


def get_decision_client() -> DecisionClient:
    """The decision provider: OpenRouter, the same gateway and key as chat.
    A second provider (TypeSafe's own API, a self-hosted /v1/systemone)
    would be another DecisionClient chosen here; nothing else changes.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise DecisionModelError("decision models run through OpenRouter: set OPENROUTER_API_KEY")
    return OpenRouterDecisionClient(api_key)


@lru_cache
def get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

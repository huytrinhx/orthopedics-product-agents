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
from functools import lru_cache
from typing import Any

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
# Whether OpenRouter's Chat Completions handles that combination is what
# backend/evals/openrouter_spike.py checks; until it has run against the
# production model this stays on. Responses-API messages carry content as a
# list of blocks, so read model output with `.text`, not `.content`.
USE_RESPONSES_API = True

# The failures that mean the gateway itself is down or misconfigured: a 5xx,
# a timeout or dropped connection (APITimeoutError is an
# APIConnectionError), or a rejected key. A 4xx about the request, or a
# model output that doesn't parse, would fail the same way on OpenAI
# direct, so those still raise.
GATEWAY_FAILURES: tuple[type[Exception], ...] = (
    openai.InternalServerError,
    openai.APIConnectionError,
    openai.AuthenticationError,
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
    return ChatOpenAI(
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


@lru_cache
def get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

"""LLM/agent-specific tracing via Langfuse Cloud: prompts, token usage,
per-node graph execution, and judge scores, all tied to one trace per chat
turn. Kept separate from otel_setup.py because it captures LLM content, not
just service metrics -- otel_setup.py covers generic service health
(request latency, error rates, dependency calls); this covers "what
actually happened for this one user question."

Cloud, not self-hosted: no PHI/compliance driver, and volume (~100-150
questions/day) is far inside the free tier, so the self-hosted stack
(web+worker+ClickHouse+Redis+MinIO) would be pure ops overhead for no
benefit. One Langfuse Cloud project covers local dev, production, and
offline evals -- LANGFUSE_TRACING_ENVIRONMENT segments them (see below)
instead of separate infra.

Every registered LangGraph workflow gets this automatically: merge the
config fragment from new_callback_handler() into the config dict passed to
astream_events()/ainvoke(), and every LLM call, tool call, and graph node
becomes a span with zero changes inside the workflow's own node functions
(langfuse.langchain.CallbackHandler hooks LangChain's callback system
directly). The Langfuse SDK itself no-ops safely (logs a warning, never
raises) when LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't set -- e.g. in
CI, which has neither -- so nothing here needs to guard against that.
"""
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from agents.state import EvalScores


def configure_langfuse() -> None:
    """Call once at process startup (backend/api/main.py's lifespan) so the
    client initializes -- and logs a clear warning if unconfigured -- up
    front rather than silently on the first chat request.
    """
    get_client()


def new_callback_handler(
    *, user_id: str, session_id: str, tags: list[str]
) -> tuple[CallbackHandler, dict]:
    """A fresh per-request handler plus the LangGraph config fragment
    (callbacks + metadata) that correlates its trace to this user/session --
    merge the returned dict into the graph's own `config`, e.g.:

        handler, langfuse_config = new_callback_handler(...)
        config = {"configurable": {...}, **langfuse_config}

    `handler.last_trace_id` is populated once the graph run completes --
    pass it to score_trace()/get_trace_url() afterward.
    """
    handler = CallbackHandler()
    config_fragment = {
        "callbacks": [handler],
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": user_id,
            "langfuse_tags": tags,
        },
    }
    return handler, config_fragment


def score_trace(trace_id: str | None, *, eval_scores: EvalScores | None, loop_count: int | None) -> None:
    """Attaches the final answer's judge scores plus the workflow's loop
    count to one trace, as separate NUMERIC scores so each is independently
    filterable/sortable in Langfuse's UI (e.g. "traces where citation < 0.6").

    Call this exactly once per trace, after the graph run finishes, with the
    *final delivered* answer's scores -- not once per internal retry. A
    workflow's own internal self-eval calls (e.g. deterministic.py's retry
    loop) still show up automatically as their own generation spans via the
    callback handler; this is the trace-level judgment, separate from that
    detail.
    """
    if not trace_id:
        return
    client = get_client()
    for axis, value in (eval_scores or {}).items():
        client.create_score(trace_id=trace_id, name=axis, value=value, data_type="NUMERIC")
    if loop_count is not None:
        client.create_score(trace_id=trace_id, name="loop_count", value=loop_count, data_type="NUMERIC")


def get_trace_url(trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    return get_client().get_trace_url(trace_id=trace_id)

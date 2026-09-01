"""Unit-level coverage for backend/observability/langfuse_setup.py that
doesn't need real Langfuse credentials or a real LLM call -- mocks the
Langfuse client itself, unlike test_chat_routes.py's end-to-end
needs_openai_key test which hits the real SDK to verify a trace actually
lands in Langfuse Cloud.
"""
from unittest.mock import MagicMock, patch

from observability.langfuse_setup import get_trace_url, new_callback_handler, score_trace


def test_score_trace_writes_one_score_per_eval_axis_plus_loop_count():
    with patch("observability.langfuse_setup.get_client") as get_client:
        client = MagicMock()
        get_client.return_value = client

        score_trace(
            "trace-123",
            eval_scores={"faithfulness": 0.9, "relevance": 0.8},
            loop_count=1,
        )

    assert client.create_score.call_count == 3
    calls_by_name = {call.kwargs["name"]: call.kwargs["value"] for call in client.create_score.call_args_list}
    assert calls_by_name == {"faithfulness": 0.9, "relevance": 0.8, "loop_count": 1}
    for call in client.create_score.call_args_list:
        assert call.kwargs["trace_id"] == "trace-123"
        assert call.kwargs["data_type"] == "NUMERIC"


def test_score_trace_omits_loop_count_score_when_none():
    with patch("observability.langfuse_setup.get_client") as get_client:
        client = MagicMock()
        get_client.return_value = client

        score_trace("trace-123", eval_scores={"faithfulness": 0.9}, loop_count=None)

    names = {call.kwargs["name"] for call in client.create_score.call_args_list}
    assert names == {"faithfulness"}


def test_score_trace_noops_without_a_trace_id():
    with patch("observability.langfuse_setup.get_client") as get_client:
        score_trace(None, eval_scores={"faithfulness": 0.9}, loop_count=2)
    get_client.assert_not_called()


def test_get_trace_url_noops_without_a_trace_id():
    with patch("observability.langfuse_setup.get_client") as get_client:
        assert get_trace_url(None) is None
    get_client.assert_not_called()


def test_new_callback_handler_config_fragment_carries_session_user_and_tags():
    with patch("observability.langfuse_setup.CallbackHandler") as handler_cls:
        handler_cls.return_value = MagicMock()
        handler, config_fragment = new_callback_handler(
            user_id="user-1", session_id="user-1:abc", tags=["deterministic"]
        )

    assert handler is handler_cls.return_value
    assert config_fragment["callbacks"] == [handler]
    assert config_fragment["metadata"] == {
        "langfuse_session_id": "user-1:abc",
        "langfuse_user_id": "user-1",
        "langfuse_tags": ["deterministic"],
    }

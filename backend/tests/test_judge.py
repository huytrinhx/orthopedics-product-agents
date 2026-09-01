"""Exercises backend/agents/judge.py. Real LLM calls, so gated behind
OPENAI_API_KEY like test_entity_extraction.py/test_embedding.py -- there's
nothing about judge_answer worth testing without a real model behind it,
its whole job is the judgment call.
"""
import os

import pytest

from agents.judge import judge_answer

needs_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="requires a real OPENAI_API_KEY"
)


@needs_openai_key
async def test_faithful_cited_answer_scores_high():
    retrieved = [
        {
            "chunk_id": "doc-1#0",
            "document_id": "doc-1",
            "text": "The MPPA100L screw has a diameter of 3.5mm and length of 20mm.",
            "score": 0.9,
        }
    ]
    answer = "The MPPA100L screw is 3.5mm in diameter and 20mm long [doc-1#0]."

    scores = await judge_answer("What are the specs of the MPPA100L screw?", retrieved, answer)

    assert scores["faithfulness"] > 0.7
    assert scores["relevance"] > 0.7
    assert scores["citation"] > 0.7


@needs_openai_key
async def test_uncited_fabricated_answer_scores_low_on_faithfulness_and_citation():
    retrieved = [
        {
            "chunk_id": "doc-1#0",
            "document_id": "doc-1",
            "text": "The MPPA100L screw has a diameter of 3.5mm and length of 20mm.",
            "score": 0.9,
        }
    ]
    # Answer invents a diameter the context doesn't support, and cites nothing.
    answer = "The MPPA100L screw is 6.0mm in diameter and comes in titanium and steel variants."

    scores = await judge_answer("What are the specs of the MPPA100L screw?", retrieved, answer)

    assert scores["faithfulness"] < 0.5
    assert scores["citation"] < 0.5


@needs_openai_key
async def test_honest_no_answer_is_not_penalized_for_faithfulness():
    answer = "The provided context doesn't include information about that."

    scores = await judge_answer("What is the torque spec for the driver?", [], answer)

    assert scores["faithfulness"] > 0.7

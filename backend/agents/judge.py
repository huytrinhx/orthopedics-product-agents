"""Shared LLM-judge module.

Scores a (query, retrieved passages, answer) triple against the same four
axes users rate in the feedback UI: faithfulness, relevance, style, citation.
Used by the inline self-eval node, the offline eval harness
(backend/evals/harness.py), and to calibrate against human feedback
(backend/api/routes/feedback.py) — all three share this module so automated
and human scores stay directly comparable.
"""
from agents.state import EvalScores, RetrievedPassage


async def judge_answer(
    query: str,
    retrieved: list[RetrievedPassage],
    answer: str,
) -> EvalScores:
    raise NotImplementedError("Implement LLM-judge scoring for the four axes")

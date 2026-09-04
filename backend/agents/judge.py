"""Shared LLM-judge module.

Scores a (query, retrieved passages, answer) triple against the same four
axes users rate in the feedback UI: faithfulness, relevance, style, citation.
Used by the inline self-eval node, the offline eval harness
(backend/evals/harness.py), and to calibrate against human feedback
(backend/api/routes/feedback.py) — all three share this module so automated
and human scores stay directly comparable.
"""
from pydantic import BaseModel, Field

from agents.state import EvalScores, RetrievedPassage
from config.llm_clients import get_chat_model

_SYSTEM_PROMPT = """You are grading an AI assistant's answer to a product-knowledge
question about orthopedic implant systems, against the context it was given.
Score each axis from 0 (worst) to 1 (best):

- faithfulness: every factual claim in the answer is actually supported by
  the provided context. Score low if the answer states anything the context
  doesn't back up, even if it happens to be true in general.
- relevance: the answer actually addresses what the question asked, not a
  related-but-different topic.
- style: the answer is clear, concise, and well-formatted (e.g. bulleted
  specs rather than a wall of prose) for a sales rep who needs the answer
  fast.
- citation: every factual claim is attributed to a source that actually
  appears in the given context, and no claim is left uncited. A claim drawn
  from a document passage needs a bracketed source id (e.g. [doc-id#3]); a
  claim drawn from a catalog/database fact (a SKU, spec, or part property --
  usually given in the context under a "catalog-facts"/"tool-result" id, or
  named as one of the SKUs listed there) doesn't need brackets, naming the
  SKU inline is the correct citation for that kind of fact. Don't score
  citation low just because an answer has no bracketed ids if every claim
  in it is actually a catalog/database fact named this way.

If the context is empty or the answer says it can't find an answer, that is
faithful and should not be penalized just for being unhelpful -- score
relevance/citation based on whether that's actually the right call given the
context provided."""


class _JudgeScores(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    style: float = Field(ge=0, le=1)
    citation: float = Field(ge=0, le=1)


def _format_context(retrieved: list[RetrievedPassage]) -> str:
    if not retrieved:
        return "(no context was retrieved)"
    return "\n\n".join(f"[{passage['chunk_id']}] {passage['text']}" for passage in retrieved)


async def judge_answer(
    query: str,
    retrieved: list[RetrievedPassage],
    answer: str,
) -> EvalScores:
    model = get_chat_model().with_structured_output(_JudgeScores)
    result = await model.ainvoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Context given to the assistant:\n{_format_context(retrieved)}\n\n"
                    f"Assistant's answer:\n{answer}"
                ),
            },
        ]
    )
    return EvalScores(
        faithfulness=result.faithfulness,
        relevance=result.relevance,
        style=result.style,
        citation=result.citation,
    )

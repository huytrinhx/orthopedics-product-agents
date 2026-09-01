"""Deterministic retrieval workflow: fixed pipeline, no agentic tool choice.

query -> resolve_synonyms -> hybrid_retrieve -> rerank -> generate -> self_eval
       -> (loop: reformulate + retry if self_eval faithfulness/relevance low)
       -> finalize

This is the baseline workflow every other architecture is compared against
in the eval harness (backend/evals/harness.py). "Fixed pipeline" means this
graph always calls vector_search (backend/agents/tools/vector_search.py,
ticket 06's hybrid vector+full-text search) in the same place every turn --
it never chooses to call graph_query instead the way the ReAct agent
(backend/agents/workflows/react_agent.py) will. synonym_resolve is a single
lookup against the whole query text, not per-extracted-entity: picking which
terms in a free-text question are worth resolving is exactly the kind of
judgment call this workflow deliberately doesn't make, so most turns are a
no-op here and that's fine for a fixed baseline.
"""
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents.judge import judge_answer
from agents.registry import register
from agents.state import BaseAgentState, RetrievedPassage
from agents.tools.synonym_resolve import synonym_resolve
from agents.tools.vector_search import vector_search
from config.llm_clients import get_chat_model

MAX_RETRIEVAL_LOOPS = 2
# Below this on faithfulness or relevance, the answer is worth one more
# retrieval attempt with a reformulated query rather than shipping as-is.
RETRY_SCORE_THRESHOLD = 0.6
# hybrid_retrieve pulls a wider candidate pool than actually goes to
# generation -- rerank (an LLM relevance pass, not just embedding
# similarity) is what narrows it down, so it needs real candidates to
# choose among.
RETRIEVE_TOP_K = 12
RERANK_TOP_N = 5


class DeterministicState(BaseAgentState, total=False):
    # `query` (from BaseAgentState) is always the user's actual question,
    # verbatim -- self_eval judges relevance against it and generate echoes
    # it back, so reformulate must never touch it. `search_query` is what
    # retrieval actually uses; it starts equal to `query` and is the only
    # thing reformulate rewrites. Both are reset per turn by the API layer
    # (backend/api/routes/chat.py) -- see its inputs dict -- since neither
    # has a reducer and would otherwise leak a prior turn's leftover value
    # in via the checkpointer.
    search_query: str
    resolved_synonyms: list[str]
    reranked: list[RetrievedPassage]
    answer: str
    citations: list[str]
    retrieval_loop_count: int


class _RerankScores(BaseModel):
    scores: list[float] = Field(
        description=(
            "One relevance score from 0 (irrelevant) to 1 (highly relevant) "
            "per passage, in the same order the passages were given."
        )
    )


class _ReformulatedQuery(BaseModel):
    query: str = Field(
        description="A rewritten search query more likely to retrieve context that answers the original question."
    )


async def resolve_synonyms(state: DeterministicState) -> dict:
    search_query = state.get("search_query") or state["query"]
    synonyms = await synonym_resolve.ainvoke({"term": search_query})
    return {"resolved_synonyms": synonyms}


async def hybrid_retrieve(state: DeterministicState) -> dict:
    search_query = state.get("search_query") or state["query"]
    search_text = " ".join([search_query, *state.get("resolved_synonyms", [])])
    results = await vector_search.ainvoke({"query": search_text, "top_k": RETRIEVE_TOP_K})
    retrieved: list[RetrievedPassage] = [
        {
            "chunk_id": result["citation"],
            "document_id": result["document_id"],
            "text": result["content"],
            "score": result["score"],
        }
        for result in results
    ]
    return {"retrieved": retrieved}


async def rerank(state: DeterministicState) -> dict:
    passages = state["retrieved"]
    if not passages:
        return {"reranked": []}
    listing = "\n\n".join(f"[{i}] {passage['text']}" for i, passage in enumerate(passages))
    model = get_chat_model().with_structured_output(_RerankScores)
    result = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Score how relevant each numbered passage is to answering "
                    "the query, 0 to 1. Return exactly one score per passage, "
                    "in the same order."
                )
            ),
            {"role": "user", "content": f"Query: {state['query']}\n\nPassages:\n{listing}"},
        ]
    )
    scored = sorted(zip(passages, result.scores), key=lambda pair: pair[1], reverse=True)
    return {"reranked": [passage for passage, _ in scored[:RERANK_TOP_N]]}


_GENERATE_SYSTEM_PROMPT = (
    "You are OrthoMate, a product-knowledge assistant for orthopedics sales "
    "reps. Answer only using the provided context passages. Cite every "
    "factual claim with its bracketed chunk id, e.g. [doc-id#0]. If the "
    "context doesn't answer the question, say so plainly rather than "
    "guessing. Prefer bulleted specs over prose paragraphs when listing "
    "measurements, SKUs, or options."
)


def _format_context(passages: list[RetrievedPassage]) -> str:
    if not passages:
        return "(no relevant context found)"
    return "\n\n".join(f"[{passage['chunk_id']}] {passage['text']}" for passage in passages)


# Matches the [doc-id#chunk-index] markers the generate prompt asks the
# model to cite with -- used to report only the citations the answer
# actually used, not every passage that happened to be in its context
# window (most of which a given answer won't end up referencing).
_CITATION_PATTERN = re.compile(r"\[([^\[\]]+#\d+)\]")


def _extract_citations(answer: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _CITATION_PATTERN.finditer(answer):
        seen.setdefault(match.group(1), None)
    return list(seen)


async def generate(state: DeterministicState) -> dict:
    passages = state.get("reranked") or []
    model = get_chat_model()
    # state["messages"]'s last entry is this turn's raw HumanMessage (the
    # API appends it before invoking, and the checkpointer restores every
    # earlier turn's messages too) -- swap it for a context-augmented
    # version so the model sees the retrieved passages, while keeping every
    # earlier turn as real conversational history. This is a *draft*: on a
    # retry, generate may run again with a reformulated search_query, and
    # only the version self_eval accepts should become permanent
    # conversation history -- see finalize below, not this node, for the
    # actual `messages` append.
    history = state["messages"][:-1]
    augmented_question = HumanMessage(
        content=f"Context:\n{_format_context(passages)}\n\nQuestion: {state['query']}"
    )
    response = await model.ainvoke(
        [SystemMessage(content=_GENERATE_SYSTEM_PROMPT), *history, augmented_question]
    )
    answer = response.content
    return {"answer": answer, "citations": _extract_citations(answer)}


async def self_eval(state: DeterministicState) -> dict:
    scores = await judge_answer(state["query"], state.get("reranked") or [], state["answer"])
    return {"eval_scores": scores}


async def reformulate(state: DeterministicState) -> dict:
    model = get_chat_model().with_structured_output(_ReformulatedQuery)
    result = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "The previous search query didn't retrieve context good "
                    "enough to answer the question well. Rewrite it to search "
                    "better -- consider synonyms, more specific product "
                    "terminology, or a narrower focus."
                )
            ),
            {"role": "user", "content": f"Original question: {state['query']}"},
        ]
    )
    return {
        "search_query": result.query,
        "retrieval_loop_count": state.get("retrieval_loop_count", 0) + 1,
    }


async def finalize(state: DeterministicState) -> dict:
    """Commits the accepted answer to permanent conversation history. A
    kept-separate step (not done in generate) so a discarded draft from an
    earlier retry attempt never ends up alongside the accepted one -- only
    the answer self_eval actually accepted should shape future turns.
    """
    return {"messages": [AIMessage(content=state["answer"])]}


def _should_retry(state: DeterministicState) -> Literal["reformulate", "finalize"]:
    scores = state.get("eval_scores") or {}
    loop_count = state.get("retrieval_loop_count", 0)
    scored_low = (
        scores.get("faithfulness", 1.0) < RETRY_SCORE_THRESHOLD
        or scores.get("relevance", 1.0) < RETRY_SCORE_THRESHOLD
    )
    if scored_low and loop_count < MAX_RETRIEVAL_LOOPS:
        return "reformulate"
    return "finalize"


def build_graph(checkpointer):
    graph = StateGraph(DeterministicState)
    graph.add_node("resolve_synonyms", resolve_synonyms)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("self_eval", self_eval)
    graph.add_node("reformulate", reformulate)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("resolve_synonyms")
    graph.add_edge("resolve_synonyms", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "self_eval")
    graph.add_conditional_edges("self_eval", _should_retry)
    graph.add_edge("reformulate", "resolve_synonyms")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


register("deterministic", build_graph)

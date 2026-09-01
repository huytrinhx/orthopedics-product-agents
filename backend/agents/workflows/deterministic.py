"""Deterministic retrieval workflow: fixed pipeline, no agentic tool choice.

query -> detect_intent -> resolve_synonyms -> hybrid_retrieve -> rerank -> generate -> self_eval
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

detect_intent (ticket 09) is the one exception to "no judgment calls": it
classifies which product system the query is about and, when it can't tell
confidently, calls interrupt() to ask rather than letting hybrid_retrieve
search unfiltered -- see its own docstring below.
"""
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agents.judge import judge_answer
from agents.registry import register
from agents.state import BaseAgentState, RetrievedPassage
from agents.tools.synonym_resolve import synonym_resolve
from agents.tools.vector_search import vector_search
from config.llm_clients import get_chat_model
from tags.repository import list_systems

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

# The real taxonomy from evals/golden_datasets/feedback-notes.csv's
# "Question Type" column (build_dataset.py carries it into mis.jsonl/
# reflex.jsonl already; intent_detection.jsonl now does too) -- fixed,
# unlike systems below, since new question types aren't something an admin
# adds through the tags UI.
QUESTION_TYPES = [
    "Specs - product characteristics",
    "Specs - system contents; SKU/ordering info",
    "Technique/procedural",
    "Pull resource",
]


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
    # Set by detect_intent, the graph's entry node. resolved_system_id is
    # what hybrid_retrieve actually filters on; resolved_system is kept
    # alongside it (rather than looked up again) for the done/eval payload
    # and because a resume answer that doesn't match any known system name
    # still gets recorded as itself, resolved_system_id just stays None.
    # resolved_question_type has no retrieval-filtering role -- it's
    # observability only, not wired into hybrid_retrieve -- but when the
    # classifier finds more than one type genuinely applies, detect_intent
    # DOES interrupt to ask which retrieval/reasoning path to take (see its
    # docstring), so this ends up a single-element list in that case too.
    # It's a list, not a single value, since a question can genuinely span
    # more than one type (e.g. feedback-notes.csv's row 25,
    # "Technique/procedural steps\nSpecs - product characteristics"),
    # matching expected_question_type's shape in
    # evals/golden_datasets/intent_detection.jsonl. Always sanitized to
    # exactly the canonical QUESTION_TYPES strings -- see
    # _normalize_question_types.
    resolved_system: str | None
    resolved_system_id: str | None
    resolved_question_type: list[str]


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


class _IntentClassification(BaseModel):
    system: str | None = Field(
        default=None,
        description=(
            "The one product system name (from the allowed list given in the "
            "prompt) the query is clearly about, or null if it doesn't "
            "clearly name or imply exactly one -- don't guess between two "
            "plausible systems."
        ),
    )
    question_type: list[str] = Field(
        default_factory=list,
        description=(
            "Zero or more of the allowed question types that apply -- most "
            "questions are exactly one, but include more than one only when "
            "the question genuinely asks about more than one kind of thing "
            "(e.g. both a technique/procedural step and a product spec in "
            "the same question). Empty if none clearly apply."
        ),
    )


def _match_option(answer: str, options: list[str]) -> str | None:
    normalized = answer.strip().lower()
    for option in options:
        if option.lower() == normalized:
            return option
    return None


def _normalize_question_types(raw: list[str]) -> list[str]:
    """Sanitizes the classifier's free-form question_type strings down to
    exactly the canonical QUESTION_TYPES values -- an LLM asked to echo a
    long fixed string back (e.g. "Specs - system contents; SKU/ordering
    info") will occasionally paraphrase, drop punctuation, or vary casing
    even when told to use the list verbatim. Matches case/whitespace-
    insensitively and drops (rather than guesses at) anything that still
    doesn't match one of the four, so a hallucinated or malformed label
    never leaks into resolved_question_type. Also dedupes and drops blanks,
    since a list output can repeat or include stray empty strings.
    """
    by_normalized = {qt.strip().lower(): qt for qt in QUESTION_TYPES}
    seen: dict[str, None] = {}
    for item in raw:
        canonical = by_normalized.get(item.strip().lower())
        if canonical is not None:
            seen.setdefault(canonical, None)
    return list(seen)


async def _classify_intent(query: str, system_names: list[str]) -> _IntentClassification:
    model = get_chat_model().with_structured_output(_IntentClassification)
    return await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Classify the sales rep's question below.\n\n"
                    f"Allowed systems: {', '.join(system_names) or '(none configured)'}\n"
                    f"Allowed question types: {', '.join(QUESTION_TYPES)}\n\n"
                    "Set system to null if the question doesn't clearly name "
                    "or imply exactly one of the allowed systems. Use the "
                    "question type strings exactly as given -- most "
                    "questions are exactly one type; only include more than "
                    "one if the question genuinely spans more than one."
                )
            ),
            {"role": "user", "content": query},
        ]
    )


async def detect_intent(state: DeterministicState) -> dict:
    """Entry node: classifies which product system and question type the
    query is about. A query that doesn't clearly name or imply one system
    (e.g. it never mentions MIS, REFLEX, or any distinguishing product
    term) pauses the graph via interrupt() for a clarifying answer, rather
    than letting hybrid_retrieve search unfiltered across every system's
    documents -- backend/api/routes/chat.py's resume_chat is the other half
    of this suspend/resume pair, and _match_option is what turns either a
    clicked option or free text back into a real system name on resume.

    Systems come from tags.list_systems() rather than a fixed enum,
    matching ticket 05's system tag -- adding a new system in the admin UI
    makes it selectable here with no code change. If none are configured
    yet (a fresh environment with nothing tagged), there's nothing to
    disambiguate against, so this skips straight past the interrupt and
    leaves resolved_system None -- retrieval runs unfiltered, the same
    baseline behavior as before this node existed.

    question_type never triggers this interrupt -- it has no
    retrieval-filtering role today, only an observability one, so an
    unclear (or multi-type) classification there is left as whatever
    _normalize_question_types resolves rather than blocking the turn on a
    second question. It's a list, not a single value, since a question can
    genuinely be more than one type (see _normalize_question_types, which
    also sanitizes the model's raw output down to the canonical
    QUESTION_TYPES strings).
    """
    systems = await list_systems()
    system_names = [s.name for s in systems]

    result = await _classify_intent(state["query"], system_names)

    if result.system is None and system_names:
        answer = interrupt(
            {"question": "Which product system is this about?", "options": system_names}
        )
        result.system = _match_option(answer, system_names)

    system_id = next((s.id for s in systems if s.name == result.system), None)
    return {
        "resolved_system": result.system,
        "resolved_system_id": str(system_id) if system_id else None,
        "resolved_question_type": _normalize_question_types(result.question_type),
    }


async def resolve_synonyms(state: DeterministicState) -> dict:
    search_query = state.get("search_query") or state["query"]
    synonyms = await synonym_resolve.ainvoke({"term": search_query})
    return {"resolved_synonyms": synonyms}


async def hybrid_retrieve(state: DeterministicState) -> dict:
    # Deliberately NOT filtered by detect_intent's resolved_system_id --
    # system_id is nullable on documents/chunks (ticket 05: tagging is
    # optional, done separately from upload), so a hard filter would
    # silently exclude every untagged document from an otherwise-answerable
    # query. resolved_system stays informational (returned in the "done"
    # payload, available for observability/future use) rather than wired
    # into retrieval -- ticket 09's acceptance criteria only need the
    # interrupt()/resume mechanic, not a filtering behavior change.
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

    Citations ride along in additional_kwargs (a standard AIMessage field,
    checkpointed like any other) rather than a separate state channel or
    table -- that's what lets ticket 10's GET /chat/threads/{id} recover
    them straight from the checkpointer on resume, the same way it recovers
    message content, instead of them only existing for the turn that's
    still streaming. Kept as the same raw "{document_id}#{chunk_index}"
    strings _extract_citations produces (not resolved to a filename here) --
    resolving is a display concern for the API layer
    (backend/api/routes/chat.py's _resolve_citations), not something every
    workflow architecture should have to know how to do.
    """
    return {
        "messages": [
            AIMessage(content=state["answer"], additional_kwargs={"citations": state["citations"]})
        ]
    }


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
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("resolve_synonyms", resolve_synonyms)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("self_eval", self_eval)
    graph.add_node("reformulate", reformulate)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "resolve_synonyms")
    graph.add_edge("resolve_synonyms", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "self_eval")
    graph.add_conditional_edges("self_eval", _should_retry)
    graph.add_edge("reformulate", "resolve_synonyms")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


register("deterministic", build_graph)

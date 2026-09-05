"""ReAct-style agentic workflow: the model chooses which retrieval tools to
call (vector_search, part_lookup, graph_query, synonym_resolve) and when to
stop, looping until it answers directly or a step budget is hit.

query -> detect_intent -> generate <-> tools -> self_eval -> finalize

Ticket 23, built 2026-09-03 after the grilling session that redirected the
original "build react_agent" ask into evolving `deterministic` instead
(tickets 20/21/22 -- see that workflow's own module docstring). This
workflow exists specifically to test the *other* half of that decision: does
letting the model choose its own tool sequence add anything beyond what
`deterministic`'s fixed resolve-then-aggregate pipeline already gets, given
the exact same tools. Deliberately NOT gated on tickets 20/21/22's fixes
landing first -- see ticket 23's own file for why that's intentional, not an
oversight.

detect_intent is imported and reused as-is from deterministic.py, not
reimplemented -- same system-disambiguation classifier, same interrupt()/
resume suspend pattern (ticket 09), so ticket 14's admin workflow picker and
ticket 09's clarification UI both work here unchanged.

`generate` here does double duty as both the reasoning step (deciding which
tool to call next) and the final-answer step (once it stops requesting
tools) -- named "generate" rather than "agent" specifically so it lines up
with backend/api/routes/chat.py's `_STREAMED_NODES` and the frontend's
per-status content reset (frontend/app/chat/page.tsx), both keyed off that
exact node name; a differently-named node here would silently reintroduce
the "answer twice" bug that fix was written for, since this loop can call
the model many times in one turn just like a deterministic retry does.

The tool-calling exchange (AIMessage-with-tool_calls / ToolMessage pairs)
lives in its own `scratchpad` channel, never the real `messages` channel --
deterministic.py's finalize node exists for exactly this reason ("only the
answer self_eval accepted should shape future turns"), and
GET /chat/threads/{id} (chat.py) renders every message in `messages` as a
chat bubble; dumping raw tool output or an empty tool-calling turn in there
would visibly break the transcript. finalize below is the only place that
appends to the real `messages`, matching deterministic.py's own convention.
"""
import asyncio
import json
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.citations import extract_citations
from agents.judge import judge_answer
from agents.registry import register
from agents.state import BaseAgentState
from agents.tools.graph_query import graph_query
from agents.tools.part_lookup import part_lookup
from agents.tools.synonym_resolve import synonym_resolve
from agents.tools.vector_search import vector_search
from agents.workflows.deterministic import detect_intent
from config.llm_clients import get_chat_model

# "Raw power" per the 2026-09-03 grilling session: generous, not tuned down
# to match deterministic's own turn budget -- the point of this workflow is
# to find the capability ceiling, not ship a cost-optimized default. Counts
# individual tool calls, not reasoning rounds (a round can request several).
MAX_TOOL_CALLS = 8

_TOOLS = [vector_search, part_lookup, graph_query, synonym_resolve]
_TOOLS_BY_NAME = {t.name: t for t in _TOOLS}

_AGENT_SYSTEM_PROMPT = (
    "You are OrthoMate, a product-knowledge assistant for orthopedics sales "
    "reps. Decide for yourself which tools to call, in what order, and when "
    "you have enough to answer -- you are not required to call every tool, "
    "or any tool at all, if the question doesn't need one.\n\n"
    "TOOLS\n"
    "- vector_search(query, top_k): full-text/semantic search over ingested "
    "product documents (brochures, surgical technique guides, inventory "
    "forms). Good for prose, procedure narrative and step order, and "
    "anything you don't already have a specific SKU or term for -- but not "
    "a substitute for part_lookup on specific part facts (thread type, "
    "material, exact SKU) even within a procedural question. A passage "
    "mentioning a screw size is not confirmation of that screw's thread "
    "type; look that up.\n"
    "- part_lookup(term, product_family): resolves a SKU or a word from a "
    "part's description directly to real catalog Part records (sku, "
    "description, thread, guidewire_spec, driver_spec, head_style, "
    "construct, indication, ...). The precise, authoritative source for a "
    "specific item's specs -- prefer this over vector_search once you know "
    "roughly what part you're looking for. Pass product_family (e.g. "
    "\"MIS\") when you know it, to avoid cross-system noise. Some parts "
    "carry an indication property naming the specific procedure/clinical "
    "intent they're for (e.g. \"Bunion (Hallux Valgus) correction\" vs. "
    "\"Fusion / Akin osteotomy\") -- when present, this is the authoritative "
    "signal for which of several similar-looking parts is actually correct.\n"
    "- graph_query(entity, relationship): relationship traversal (e.g. "
    "REQUIRES_TOOL, DIFFERENTIATES_FROM) for a specific SKU or tray name -- "
    "use when the question is about how two things relate, not just one "
    "thing's own specs.\n"
    "- synonym_resolve(term): canonical aliases for a term (e.g. \"wire\" -> "
    "\"guidepin\") -- call this on a term before part_lookup/vector_search "
    "if you suspect the rep's wording differs from the catalog's.\n\n"
    "ANSWERING\n"
    "- Cite a claim drawn from vector_search with its bracketed chunk id "
    "exactly as given, e.g. [doc-id#0]. A claim drawn from part_lookup/"
    "graph_query doesn't need a bracketed citation -- name the SKU inline "
    "instead, since it's already a direct database fact.\n"
    "- If tool results disagree on a specific value for the same item, "
    "trust part_lookup/graph_query (the parts database) over vector_search "
    "(passage text, which can carry OCR/extraction errors).\n"
    "- Don't answer beyond what your tool calls actually returned -- if "
    "you can't find something after a reasonable search, say so plainly "
    "rather than guessing.\n"
    "- When the question asks about the difference between products, or "
    "more than one candidate part could plausibly answer it (e.g. two "
    "screws with the same diameter but different thread types), don't "
    "settle for the first plausible part_lookup match. Check each "
    "candidate's construct/thread/indication, and if the question implies "
    "a specific procedure or clinical intent, call vector_search on the "
    "surgical technique/procedure guide if indication isn't already "
    "available -- match the candidate whose actual use fits, don't guess "
    "from whichever result happened to come back first.\n"
    "- Never state a screw's thread type (Full Thread/FT vs. Partial "
    "Thread/PT) from inference, from pattern-matching the query's own "
    "wording, or from an incomplete passage snippet -- always verify it "
    "with part_lookup's thread/indication properties before including it "
    "in your answer, even for a procedural/setup question where "
    "vector_search alone looks sufficient. If the query states a thread "
    "type for one specific screw (e.g. \"a 3.0 PT screw for their Akin\"), "
    "that label applies to that one screw only -- don't extend it to "
    "other screws in the same answer without checking each one "
    "independently; a different screw in the same procedure is often the "
    "opposite thread type.\n"
    "- Prefer bulleted specs over prose paragraphs when listing "
    "measurements, SKUs, or options. Compress a contiguous, evenly-spaced "
    "numeric range into one line with its increment rather than "
    "enumerating every value."
)


class ReactAgentState(BaseAgentState, total=False):
    search_query: str
    resolved_system: str | None
    resolved_system_id: str | None
    resolved_question_type: list[str]
    # The tool-calling loop's own transcript -- see module docstring for why
    # this is a separate channel from `messages`, never merged into it
    # except by finalize appending exactly one clean AIMessage.
    scratchpad: Annotated[list, add_messages]
    tool_calls_made: int
    answer: str
    citations: list[str]


async def generate(state: ReactAgentState) -> dict:
    """Both the reasoning step (may request tools) and, once it stops
    requesting them, the final-answer step -- see module docstring for why
    this isn't split into two differently-named nodes.

    Tools are only bound while budget remains; once MAX_TOOL_CALLS is spent,
    the same node is called without tools bound, which forces a text-only
    response synthesizing whatever the scratchpad already holds instead of
    silently truncating mid-reasoning with an empty answer.
    """
    existing = state.get("scratchpad") or []
    seed: list = []
    if not existing:
        system = state.get("resolved_system")
        context_note = f" The rep's question has been classified as about the {system!r} system." if system else ""
        seed = [
            SystemMessage(content=_AGENT_SYSTEM_PROMPT + context_note),
            HumanMessage(content=state["query"]),
        ]

    model = get_chat_model()
    budget_left = state.get("tool_calls_made", 0) < MAX_TOOL_CALLS
    bound = model.bind_tools(_TOOLS) if budget_left else model
    response = await bound.ainvoke(existing + seed)

    return {"scratchpad": [*seed, response]}


async def call_tools(state: ReactAgentState) -> dict:
    last = state["scratchpad"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    async def run_one(call: dict):
        tool = _TOOLS_BY_NAME.get(call["name"])
        if tool is None:
            return f"Unknown tool: {call['name']!r}"
        try:
            return await tool.ainvoke(call["args"])
        except Exception as exc:  # noqa: BLE001 - report the failure to the model, don't crash the turn
            return f"Tool call failed: {exc}"

    results = await asyncio.gather(*(run_one(call) for call in tool_calls))
    tool_messages = [
        ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
        for call, result in zip(tool_calls, results)
    ]
    return {
        "scratchpad": tool_messages,
        "tool_calls_made": state.get("tool_calls_made", 0) + len(tool_calls),
    }


def _should_call_tools(state: ReactAgentState) -> Literal["tools", "self_eval"]:
    last = state["scratchpad"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "self_eval"


def _passages_from_scratchpad(scratchpad: list) -> list[dict]:
    """Synthesizes RetrievedPassage-shaped dicts from this turn's tool
    results so self_eval has something concrete to judge faithfulness
    against -- deterministic.py's judge_answer call expects that shape, and
    a ReAct loop's "context" is whatever its tool calls actually returned,
    not a fixed `reranked` list. Tool output is truncated defensively
    (arbitrary tool results, unlike deterministic's bounded passage text).
    """
    passages = []
    for i, msg in enumerate(scratchpad):
        if isinstance(msg, ToolMessage):
            passages.append(
                {
                    "chunk_id": f"tool-call-{i}",
                    "document_id": "tool-result",
                    "text": str(msg.content)[:4000],
                    "score": 1.0,
                    "document_type": None,
                }
            )
    return passages


async def self_eval(state: ReactAgentState) -> dict:
    passages = _passages_from_scratchpad(state.get("scratchpad") or [])
    answer = state["scratchpad"][-1].content
    scores = await judge_answer(state["query"], passages, answer)
    return {"eval_scores": scores}


async def finalize(state: ReactAgentState) -> dict:
    """The only place `messages` gets an AI turn appended -- see module
    docstring. Mirrors deterministic.py's finalize exactly (same citation
    format, same additional_kwargs convention) so ticket 10's transcript
    read and ticket 11's per-message feedback both work unchanged here.
    """
    answer = state["scratchpad"][-1].content
    citations = extract_citations(answer)
    return {
        "answer": answer,
        "citations": citations,
        "messages": [AIMessage(content=answer, additional_kwargs={"citations": citations})],
    }


def build_graph(checkpointer):
    graph = StateGraph(ReactAgentState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("generate", generate)
    graph.add_node("tools", call_tools)
    graph.add_node("self_eval", self_eval)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "generate")
    graph.add_conditional_edges("generate", _should_call_tools)
    graph.add_edge("tools", "generate")
    graph.add_edge("self_eval", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


register("react_agent", build_graph, functional=True)

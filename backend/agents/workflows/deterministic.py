"""Deterministic retrieval workflow: fixed pipeline, no agentic tool choice.

query -> detect_intent -> resolve_synonyms -> resolve_query_entities
       -> hybrid_retrieve -> rerank -> resolve_skus -> aggregate_facts
       -> generate -> self_eval
       -> (loop: reformulate + retry if self_eval faithfulness/relevance low)
       -> finalize

This is the baseline workflow every other architecture is compared against.
`react_agent` (ticket 23, backend/agents/workflows/react_agent.py) is a
deliberately separate, actually-agentic workflow -- decided in the
2026-09-03 grilling session not to fold tool-choice into this graph, since
resolve_skus/aggregate_facts below is fixed control flow with a
question-type branch, not a case where the model needs to decide what to
do next.

"Fixed pipeline" means this graph always calls vector_search
(backend/agents/tools/vector_search.py, ticket 06) in the same place every
turn for prose/context, and always calls part_lookup
(backend/agents/tools/part_lookup.py, ticket 20) to ground spec/SKU/pairing
facts directly in the graph's Part-node properties -- it never chooses
between them. resolve_synonyms and resolve_skus both extract candidate
terms from text (agents/tools/term_extraction.py) and resolve each one
individually against the graph (ticket 21) -- a whole free-text sentence
essentially never matches a Term or Part node as a literal string, so
per-term lookup is what makes either of these nodes do anything.

detect_intent (ticket 09) is the one exception to "no judgment calls": it
classifies which product system the query is about and, when it can't tell
confidently, calls interrupt() to ask rather than letting hybrid_retrieve
search unfiltered -- see its own docstring below.
"""
import asyncio
import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agents.citations import extract_citations
from agents.judge import judge_answer
from agents.registry import register
from agents.state import BaseAgentState, RetrievedPassage
from agents.tools.part_lookup import part_lookup
from agents.tools.synonym_resolve import synonym_resolve
from agents.tools.term_extraction import extract_candidate_terms, with_singular_variants
from agents.tools.vector_search import vector_search
from config.llm_clients import get_chat_model
from tags.repository import list_systems

# Neo4j round-trips are cheap individually but resolve_skus can extract
# dozens of candidate terms from a long procedural passage -- capped so one
# turn can't fire an unbounded number of concurrent graph queries.
MAX_TERMS_PER_TURN = 40

logger = logging.getLogger(__name__)

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
    # Set by resolve_query_entities (ticket 24) -- candidate Part nodes
    # resolved from the raw query alone, before retrieval runs. Used both
    # to sharpen hybrid_retrieve's vector query and (reused, not recomputed)
    # as resolve_skus's starting point for the spec/SKU path.
    query_resolved_parts: list[dict]
    reranked: list[RetrievedPassage]
    # Set by resolve_skus/aggregate_facts (ticket 22) -- resolved_parts is
    # every candidate Part node matched from this turn's extracted terms,
    # deduped by SKU; aggregated_facts is that same data formatted into
    # generate's context. Multiple candidates are never disambiguated down
    # to one here -- see resolve_skus's docstring for why.
    resolved_parts: list[dict]
    aggregated_facts: str
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
                    "or imply exactly one of the allowed systems -- when "
                    "genuinely unclear, always prefer null over guessing, "
                    "even if only one system is configured. Only once the "
                    "question does clearly identify a system: if several "
                    "allowed systems share a common family name (e.g. a "
                    "generic name alongside more specific sub-systems under "
                    "it), pick the most specific one that matches rather "
                    "than the generic family name, and don't switch to a "
                    "different sub-system just because of incidental word "
                    "overlap (a question about a screw, wire, or guidepin "
                    "describes the implant/consumable system, not a "
                    "separate powered-instrument system, unless it "
                    "explicitly asks about the powered tool itself). Use "
                    "the question type strings exactly as given -- most "
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
    if not system_names:
        # list_systems() only returns tags currently attached to at least
        # one document (tags/repository.py's _list_tags) -- an empty result
        # is the expected "nothing tagged yet" case on a fresh environment,
        # but on a populated one (documents ARE tagged) it's unexpected and
        # silently disables detect_intent's disambiguation guardrail for
        # this turn (see the eval reflection, 2026-09-03: this happened
        # intermittently against a DB known to have tagged documents).
        # Logged, not raised, since the graph's existing fallback (skip the
        # interrupt, run retrieval unfiltered) is still the right behavior
        # either way -- this just makes the unexpected case observable.
        logger.warning(
            "detect_intent: list_systems() returned no systems for query=%r "
            "-- disambiguation guardrail skipped this turn",
            state["query"],
        )

    result = await _classify_intent(state["query"], system_names)
    if result.system is not None and result.system.strip().lower() in ("null", "none", ""):
        # Structured output can put the literal word "null"/"none" *into*
        # the string field instead of actually leaving it unset -- seen for
        # real in the 2026-09-03 ticket 20/21/22 eval run (Q10), where this
        # then propagated as a bogus ProductFamily scope and silently
        # zeroed out resolve_skus's graph matches for the whole turn.
        result.system = None

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
    """Ticket 21: resolves synonyms per extracted term, not the whole query
    string. synonym_resolve looks up an exact Term-node name match (e.g.
    "wire" -[:ALIAS_OF]-> "guidepin") -- a multi-word sentence essentially
    never matches one of those literally, so the old whole-query call was a
    no-op on nearly every real turn despite the graph having the right data
    (confirmed directly against Neo4j in the 2026-09-03 eval reflection).
    """
    search_query = state.get("search_query") or state["query"]
    terms = with_singular_variants(extract_candidate_terms(search_query))[:MAX_TERMS_PER_TURN]
    results = await asyncio.gather(*(synonym_resolve.ainvoke({"term": term}) for term in terms))
    seen: dict[str, None] = {}
    for synonyms in results:
        for synonym in synonyms:
            seen.setdefault(synonym, None)
    return {"resolved_synonyms": list(seen)}


async def resolve_query_entities(state: DeterministicState) -> dict:
    """Ticket 24: resolves the raw query directly to candidate Part nodes
    via the graph, BEFORE vector_search runs, so hybrid_retrieve can search
    on the graph's own canonical descriptions/SKUs, not just the rep's raw
    wording -- e.g. once "wire" resolves to a real guidepin SKU, the vector
    query also includes "GUIDEPIN MIS 3.5 PT 1.4 X 150MM", which is far
    closer to what the actual inventory-table chunk text looks like than
    the word "wire" ever is.

    Also reused by resolve_skus below for the spec/SKU path, which needs
    the identical lookup -- this node's result is authoritative for "what
    does the query itself, alone, resolve to"; resolve_skus's own job is
    folding in whatever the *procedural* path's retrieved passages
    additionally surface, which this node can't see yet (it runs before
    retrieval).
    """
    terms = with_singular_variants(
        [*extract_candidate_terms(state["query"]), *state.get("resolved_synonyms", [])]
    )[:MAX_TERMS_PER_TURN]
    family = state.get("resolved_system")
    results = await asyncio.gather(
        *(part_lookup.ainvoke({"term": term, "product_family": family}) for term in terms)
    )
    matched: dict[str, dict] = {}
    for parts in results:
        for part in parts:
            sku = part.get("sku")
            if sku:
                matched.setdefault(sku, part)
    return {"query_resolved_parts": list(matched.values())}


def _entity_search_addendum(resolved_parts: list[dict], limit: int = 5) -> str:
    """A handful of resolved parts' own descriptions, appended to the
    vector search text -- capped, since this is meant to sharpen the query
    with a few concrete, canonical terms, not flood it with the entire
    resolved set (which can run to dozens of parts for a broad term)."""
    return " ".join(part.get("description", "") for part in resolved_parts[:limit] if part.get("description"))


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
    entity_addendum = _entity_search_addendum(state.get("query_resolved_parts") or [])
    search_text = " ".join(
        filter(None, [search_query, *state.get("resolved_synonyms", []), entity_addendum])
    )
    results = await vector_search.ainvoke({"query": search_text, "top_k": RETRIEVE_TOP_K})
    retrieved: list[RetrievedPassage] = [
        {
            "chunk_id": result["citation"],
            "document_id": result["document_id"],
            "text": result["content"],
            "score": result["score"],
            "document_type": result.get("document_type"),
        }
        for result in results
    ]
    return {"retrieved": retrieved}


# doctype-hierarchy.csv's priority order, expressed in this system's real
# document_type names (tags/repository.py's list_document_types()), not the
# CSV's own generic wording ("Inventory Control Form" here is "Inventory
# Control", etc.). That CSV's own Priority 1 ("Master Item File") isn't a
# document at all -- it's the graph, already covered by resolve_skus/
# aggregate_facts (ticket 20) -- so this ranks what's left: the P2-P4
# document-chunk tiers, as a rerank tiebreak. Matched by substring, since a
# real document_type name and the CSV's wording don't line up exactly.
_DOCTYPE_PRIORITY: dict[str, list[str]] = {
    "Specs - system contents; SKU/ordering info": [
        "Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique",
    ],
    "Specs - product characteristics": [
        "Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique",
    ],
    "Technique/procedural": ["Surgical Technique", "Setup Guide", "Launch Presentation"],
}
# Additive score bonus per priority rank -- small relative to rerank's 0-1
# LLM relevance scores, so it breaks ties/favors a more-authoritative source
# on close calls rather than overriding a much more semantically relevant
# passage from a lower-tier doctype.
_DOCTYPE_BONUS_STEP = 0.05


def _doctype_priority_bonus(question_types: list[str], document_type: str | None) -> float:
    if not document_type:
        return 0.0
    for question_type in question_types:
        order = _DOCTYPE_PRIORITY.get(question_type)
        if not order:
            continue
        for rank, name in enumerate(order):
            if name.lower() in document_type.lower():
                return (len(order) - rank) * _DOCTYPE_BONUS_STEP
        return 0.0  # this question type has a defined order and this doctype just isn't ranked in it
    return 0.0


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
    question_types = state.get("resolved_question_type") or []
    # score is overwritten with the combined (LLM relevance + doctype
    # priority bonus) value, not left at hybrid_retrieve's original RRF
    # fusion score -- that's the number that actually decided this passage's
    # rank, and a trace/debugging view showing the old, pre-rerank score
    # here would misrepresent why a given passage made the cut.
    combined = [
        {**passage, "score": score + _doctype_priority_bonus(question_types, passage.get("document_type"))}
        for passage, score in zip(passages, result.scores)
    ]
    scored = sorted(combined, key=lambda passage: passage["score"], reverse=True)
    return {"reranked": scored[:RERANK_TOP_N]}


async def resolve_skus(state: DeterministicState) -> dict:
    """Ticket 22: resolves this turn to candidate Part nodes so generate
    can ground spec/SKU/pairing facts in the graph (ticket 20) instead of
    vector prose alone. Where the candidate terms come from depends on
    question_type -- the two paths converge on the same resolve-then-
    aggregate mechanism, just fed different source text:

    - "Technique/procedural": both the *retrieved* passage text (e.g. an
      inventory-table chunk naming exact SKUs next to the step that uses
      them) and the rep's own query. Originally passage-only, but the
      2026-09-03 eval reflection found a real gap that way: a procedural
      question can name specifics (a screw size, a thread-type word) that
      the reranked top-5 doesn't happen to surface a matching SKU chunk
      for, so resolve_skus had nothing to resolve regardless of how good
      the resolution mechanism itself was -- e.g. "a 4.0 and 3.5 screw for
      their bunion" naming exactly the sizes needed, with the Full-Thread
      inventory chunk not making that turn's top-5.
    - Everything else (spec/SKU/pairing lookups): the query alone -- no
      passage text, since these questions are usually answered by one or
      two specific Parts, not the broader context a procedural walkthrough
      needs.

    Either way, resolved_synonyms (already computed this turn by
    resolve_synonyms, above) is folded in for free -- e.g. a query that
    says "wire" already resolved "guidepin" as a synonym, which is the term
    that actually substring-matches Part descriptions.

    Multiple candidate Parts for one term are never narrowed to one here --
    all matches are deduped by SKU and handed to generate labeled, which
    decides what the question actually needs. Disambiguating up front (or
    interrupt()-ing like detect_intent does for system ambiguity) would
    fire on completely ordinary questions, since one query legitimately
    matching several real SKUs is the common case, not the exception.

    Ticket 24: the query-alone lookup is no longer redone here -- it's
    exactly what resolve_query_entities (which now runs earlier, so
    hybrid_retrieve can use it too) already computed. Seeding from that
    result rather than re-querying the graph means the spec/SKU path (the
    common case) costs zero additional Neo4j round-trips here; only the
    procedural path still does fresh lookups, against passage text
    resolve_query_entities couldn't have seen yet (it runs before
    retrieval).
    """
    question_types = state.get("resolved_question_type") or []
    matched: dict[str, dict] = {
        part["sku"]: part for part in (state.get("query_resolved_parts") or []) if part.get("sku")
    }

    if "Technique/procedural" in question_types:
        passage_text = " ".join(passage["text"] for passage in (state.get("reranked") or []))
        raw_terms = [*extract_candidate_terms(passage_text), *state.get("resolved_synonyms", [])]
        terms = with_singular_variants(raw_terms)[:MAX_TERMS_PER_TURN]
        # find_parts (retrieval/graph_client.py) normalizes a Postgres
        # systems-tag name like "MIS - Foot Recon" down to the Neo4j
        # ProductFamily name "MIS" itself -- passed straight through here.
        family = state.get("resolved_system")
        results = await asyncio.gather(
            *(part_lookup.ainvoke({"term": term, "product_family": family}) for term in terms)
        )
        for parts in results:
            for part in parts:
                sku = part.get("sku")
                if sku:
                    matched.setdefault(sku, part)

    return {"resolved_parts": list(matched.values())}


def _format_part(part: dict) -> str:
    sku = part.get("sku", "?")
    description = part.get("description", "")
    extra = ", ".join(
        f"{key}={value}"
        for key, value in part.items()
        if key not in ("sku", "description") and value not in (None, "")
    )
    line = f"- {sku}: {description}"
    return f"{line} ({extra})" if extra else line


async def aggregate_facts(state: DeterministicState) -> dict:
    """Formats resolve_skus's matched Part nodes into a context block
    generate treats as ground truth (see _GENERATE_SYSTEM_PROMPT) -- a
    separate node from resolve_skus, matching how rerank is kept separate
    from hybrid_retrieve, so each step's output stays independently
    inspectable (e.g. in a pipeline trace) rather than one node doing both
    the graph query and the formatting.
    """
    parts = state.get("resolved_parts") or []
    return {"aggregated_facts": "\n".join(_format_part(part) for part in parts)}


# Split into clearly labeled sections (rather than one long paragraph) after
# the 2026-09-03 eval reflection found the FORMATTING rules -- especially
# range-compression -- getting diluted once SOURCES grew long explaining the
# new known-catalog-facts channel (ticket 22). A model default-prioritizes
# instructions it can't tell apart from surrounding prose; a labeled
# section is a stronger signal than one more sentence at the end of a
# paragraph. Formatting rules stay short and imperative for the same reason.
_GENERATE_SYSTEM_PROMPT = (
    "You are OrthoMate, a product-knowledge assistant for orthopedics sales reps.\n\n"
    "SOURCES\n"
    "- Answer using the provided context passages and known catalog facts.\n"
    "- The known catalog facts are a *cross-check* for specific named items "
    "(an exact SKU, a dimension, a material, a thread type) -- they come "
    "from a term-matching lookup that's precise but not exhaustive, so they "
    "can be an incomplete, arbitrary slice of a larger family (e.g. only "
    "some of a product's screw sizes). Never treat their absence, or a "
    "short list of them, as proof something doesn't exist or that a full "
    "catalog listing is now complete -- for a question asking what's "
    "included in a set/system, the context passages (which include actual "
    "inventory-table chunks) are the more complete source; use the catalog "
    "facts there only to verify or correct specific values the passage "
    "gets wrong, not to replace the passage's coverage.\n"
    "- Cite every factual claim drawn from a context passage with its "
    "bracketed chunk id, e.g. [doc-id#0]; a claim drawn from the known "
    "catalog facts doesn't need a bracketed citation -- name the SKU "
    "inline instead, since it's already a direct database fact, not a "
    "passage reference.\n"
    "- If the known catalog facts and a context passage disagree on a "
    "specific value for the *same* item, trust the catalog facts -- "
    "they're pulled directly from the parts database, while a passage came "
    "through OCR/text extraction and can carry its own errors.\n"
    "- Don't answer beyond what the context passages and known catalog "
    "facts actually support -- if a specific item asked about isn't among "
    "either, say so plainly rather than guessing, and, when the known "
    "catalog facts include real nearby alternatives (e.g. adjacent sizes "
    "of the same part family), offer those instead of just stopping at "
    "'not found'.\n"
    "- When the question asks about the difference between products, or "
    "more than one candidate part could plausibly answer it (e.g. two "
    "screws with the same diameter but different thread types), don't "
    "settle for whichever one happens to be most prominent in the catalog "
    "facts or context. Check each candidate's construct/thread/indication "
    "properties and, if the question implies a specific procedure or "
    "clinical intent (a bunion vs. a fusion/Akin, for example), match the "
    "candidate whose indication actually fits that intent -- don't guess "
    "from incidental ordering.\n\n"
    "FORMATTING\n"
    "- ALWAYS compress a contiguous, evenly-spaced numeric list into a "
    "single range with its increment -- e.g. lengths 20mm, 22mm, 24mm ... "
    "66mm (every 2mm) become \"20-66mm, every 2mm\". Never enumerate more "
    "than a few individual values in a row when they form a clean range. "
    "Only break a range into individual lines when the values aren't "
    "evenly spaced or the reader needs to pick exactly one item (e.g. a "
    "specific SKU).\n"
    "- Prefer bulleted specs over prose paragraphs when listing "
    "measurements, SKUs, or options."
)

# Appended to _GENERATE_SYSTEM_PROMPT when detect_intent classified the
# turn as (at least partly) "Technique/procedural" -- a spec-lookup answer
# reads fine grouped by part category, but a procedural one needs the
# sequence a rep would actually follow at the table, not a category dump.
_PROCEDURAL_FORMAT_ADDENDUM = (
    " This question is a procedural/technique walkthrough: present the "
    "answer as a single numbered list, in the actual sequential order the "
    "steps would be performed, not grouped by part category. Skip a "
    "restating summary at the start or a generic closing sentence -- go "
    "straight into the numbered steps, with at most one short setup "
    "sentence first if genuinely needed."
)


def _format_context(passages: list[RetrievedPassage]) -> str:
    if not passages:
        return "(no relevant context found)"
    return "\n\n".join(f"[{passage['chunk_id']}] {passage['text']}" for passage in passages)


def _format_known_facts(aggregated_facts: str) -> str:
    return aggregated_facts if aggregated_facts else "(no catalog facts matched this question)"


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
        content=(
            f"Known catalog facts:\n{_format_known_facts(state.get('aggregated_facts', ''))}"
            f"\n\nContext:\n{_format_context(passages)}\n\nQuestion: {state['query']}"
        )
    )
    system_prompt = _GENERATE_SYSTEM_PROMPT
    if "Technique/procedural" in (state.get("resolved_question_type") or []):
        system_prompt += _PROCEDURAL_FORMAT_ADDENDUM
    response = await model.ainvoke(
        [SystemMessage(content=system_prompt), *history, augmented_question]
    )
    answer = response.content
    return {"answer": answer, "citations": extract_citations(answer)}


async def self_eval(state: DeterministicState) -> dict:
    """Judges the draft answer against everything generate was actually
    allowed to use -- the reranked passages *and* the aggregated catalog
    facts. Omitting the latter was a real bug caught in the ticket 20/21/22
    eval run: a claim correctly grounded in aggregated_facts but absent
    from (or contradicting) a vector passage -- exactly the graph-grounding
    ticket 20 exists for -- read as "unfaithful" to a judge that only ever
    saw the passages, penalizing the fix it was meant to verify.
    """
    passages = list(state.get("reranked") or [])
    aggregated_facts = state.get("aggregated_facts")
    if aggregated_facts:
        passages.append(
            {
                "chunk_id": "catalog-facts",
                "document_id": "catalog",
                "text": aggregated_facts,
                "score": 1.0,
                "document_type": None,
            }
        )
    scores = await judge_answer(state["query"], passages, state["answer"])
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
    strings agents/citations.py's extract_citations produces (not resolved to a filename here) --
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
    graph.add_node("resolve_query_entities", resolve_query_entities)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("resolve_skus", resolve_skus)
    graph.add_node("aggregate_facts", aggregate_facts)
    graph.add_node("generate", generate)
    graph.add_node("self_eval", self_eval)
    graph.add_node("reformulate", reformulate)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "resolve_synonyms")
    graph.add_edge("resolve_synonyms", "resolve_query_entities")
    graph.add_edge("resolve_query_entities", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "resolve_skus")
    graph.add_edge("resolve_skus", "aggregate_facts")
    graph.add_edge("aggregate_facts", "generate")
    graph.add_edge("generate", "self_eval")
    graph.add_conditional_edges("self_eval", _should_retry)
    graph.add_edge("reformulate", "resolve_synonyms")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


register("deterministic", build_graph)

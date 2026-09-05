"""Deterministic retrieval workflow: fixed pipeline, no agentic tool choice.

query -> detect_intent -> resolve_synonyms
       -> (if a term is genuinely ambiguous between canonical concepts:
           request_clarification, then back to resolve_synonyms)
       -> hybrid_retrieve -> rerank -> resolve_skus -> aggregate_facts
       -> generate -> self_eval
       -> (if faithfulness/relevance score low and no clarification asked
           yet this turn: request_clarification, then back to
           resolve_synonyms)
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
per-term lookup is what makes either of these nodes do anything. Both
hybrid_retrieve and resolve_skus retrieve/resolve against
resolved_canonical_terms (resolve_synonyms' output), not the rep's raw
wording alone -- see resolve_synonyms' own docstring for why canonical
terms specifically.

Three judgment-call exceptions to "no judgment calls," all using the same
interrupt()/resume suspend pattern (ticket 09) so the rep answers directly
rather than the graph silently guessing -- all three route through the
same request_clarification node and share one clarification_rounds budget
per turn (see its own docstring):
- detect_intent classifies which product system the query is about and,
  when it can't tell confidently, calls interrupt() to ask rather than
  letting hybrid_retrieve search unfiltered across every system's
  documents -- see its own docstring below.
- resolve_synonyms/_should_clarify_synonyms (2026-09-05): when a single
  extracted word matches more than one distinct canonical concept (the
  rep's own word is ambiguous, not just "the query mentions several
  things"), the graph asks which one they mean before ever running
  retrieval on a guess.
- self_eval/_should_clarify (redesigned 2026-09-04, replacing an earlier
  silent reformulate-and-retry loop): when the draft answer scores low on
  faithfulness/relevance, the graph pauses and asks the rep a specific
  clarifying question instead of having an LLM guess a better search query
  with no new information.

hybrid_retrieve also hard-filters its retrieval pool by document type
(2026-09-04) based on the classified question type (_DOCTYPE_PRIORITY,
keyed by agents/question_types.py's canonical slugs), rather than only
softly re-ranking afterward -- found live comparing an answer against its
golden expected answer that a technique_procedural question's real
step-by-step narrative chunks were being crowded out of the retrieval pool
entirely by inventory-table/marketing chunks that happened to rank higher
on raw semantic+keyword similarity; filtering the *retrieval* step itself,
not just the rerank tiebreak, is what actually keeps them out of
contention. The filter always still admits untagged chunks (tagging is
optional, ticket 05) -- see RetrievalFilters' docstring.

rerank scores that retrieval pool with BM25 (a classic keyword-overlap
ranking, 2026-09-05, replacing an LLM relevance-scoring pass -- see its own
docstring) plus the same doctype-priority bonus, so picking among an
already-retrieved candidate pool costs no model round-trip.
"""
import asyncio
import logging
import math
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agents.citations import extract_citations
from agents.judge import judge_answer
from agents.question_types import QUESTION_TYPE_NAMES, QUESTION_TYPES
from agents.registry import register
from agents.state import BaseAgentState, RetrievedPassage
from agents.tools.part_lookup import part_lookup
from agents.tools.term_extraction import extract_candidate_terms, with_singular_variants
from agents.tools.vector_search import vector_search
from config.llm_clients import get_chat_model
from retrieval.graph_client import get_graph_client
from tags.repository import TagRecord, list_document_types, list_systems

# Neo4j round-trips are cheap individually but resolve_skus can extract
# dozens of candidate terms from a long procedural passage -- capped so one
# turn can't fire an unbounded number of concurrent graph queries.
MAX_TERMS_PER_TURN = 40

logger = logging.getLogger(__name__)

# Below this on faithfulness or relevance, the answer is worth pausing to
# ask the rep a clarifying question rather than shipping as-is -- see
# request_clarification/_should_clarify below.
CLARIFICATION_SCORE_THRESHOLD = 0.6
# hybrid_retrieve pulls a wider candidate pool than actually goes to
# generation -- rerank (a BM25 keyword-relevance pass over that pool, plus
# a doctype-priority bonus -- no LLM call, see rerank's own docstring) is
# what narrows it down, so it needs real candidates to choose among.
RETRIEVE_TOP_K = 12
RERANK_TOP_N = 5

# BM25's standard tuning constants (Okapi BM25) -- k1 controls term-frequency
# saturation, b controls document-length normalization strength. 1.5/0.75
# are the conventional defaults, not tuned against this corpus specifically.
_BM25_K1 = 1.5
_BM25_B = 0.75


class DeterministicState(BaseAgentState, total=False):
    # `query` (from BaseAgentState) is always the user's actual question,
    # verbatim -- self_eval judges relevance against it and generate echoes
    # it back, so nothing past detect_intent may touch it. `search_query` is
    # what retrieval actually uses; it starts equal to `query` and is the
    # only thing request_clarification rewrites (appending the rep's reply).
    # Both search_query and the clarification fields below are reset per
    # turn by the API layer (backend/api/routes/chat.py) -- see its inputs
    # dict -- since none has a reducer and would otherwise leak a prior
    # turn's leftover value in via the checkpointer.
    search_query: str
    # Set by resolve_synonyms (2026-09-05, replacing resolved_synonyms) --
    # see that node's own docstring for why canonical terms specifically,
    # not a flat canonical+alias mix. synonym_ambiguity is only ever
    # non-None for the one turn between resolve_synonyms detecting a
    # genuinely ambiguous term and request_clarification resolving it (see
    # _should_clarify_synonyms) -- not meant to persist as a fact about the
    # turn the way resolved_canonical_terms does.
    resolved_canonical_terms: list[str]
    synonym_ambiguity: dict[str, list[str]] | None
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
    # Set by request_clarification (2026-09-04) when self_eval scores the
    # draft answer low -- clarification_reply is the rep's answer to the
    # question it asked (folded into generate's prompt so the model can
    # actually use it, not just into search_query for retrieval);
    # clarification_rounds caps this to once per turn -- see _should_clarify.
    clarification_reply: str | None
    clarification_rounds: int
    # Set by detect_intent, the graph's entry node. resolved_system_id is
    # what hybrid_retrieve actually filters on; resolved_system is kept
    # alongside it (rather than looked up again) for the done/eval payload
    # and because a resume answer that doesn't match any known system name
    # still gets recorded as itself, resolved_system_id just stays None.
    # resolved_question_type drives hybrid_retrieve's document-type filter
    # (_DOCTYPE_PRIORITY, 2026-09-04) in addition to its original
    # observability role. It's a list, not a single value, since a question
    # can genuinely span more than one type (e.g. both a procedural step
    # and a product spec in the same question) -- detect_intent does NOT
    # interrupt on a multi-type classification, only on system ambiguity
    # (see its own docstring). Always one of agents/question_types.py's
    # QUESTION_TYPE_NAMES, enforced by _IntentClassification's Literal
    # field -- not a free-form string.
    resolved_system: str | None
    resolved_system_id: str | None
    resolved_question_type: list[str]


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
    # A Literal built from QUESTION_TYPE_NAMES, not a bare `list[str]` --
    # 2026-09-05: with the taxonomy canonicalized to short slugs, the
    # structured-output schema itself constrains the model to a real value
    # (OpenAI's structured outputs enforce the enum server-side), which is
    # what made the old free-form-string normalizer (_normalize_question_
    # types, matching case/whitespace-insensitively against a hand-typed
    # verbose CSV phrase like "Specs - system contents; SKU/ordering info")
    # redundant -- removed rather than kept as unused dead code.
    question_type: list[Literal[*QUESTION_TYPE_NAMES]] = Field(
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


async def _classify_intent(query: str, system_names: list[str]) -> _IntentClassification:
    model = get_chat_model().with_structured_output(_IntentClassification)
    question_type_listing = "\n".join(f"- {qt.name}: {qt.description}" for qt in QUESTION_TYPES)
    return await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Classify the sales rep's question below.\n\n"
                    f"Allowed systems: {', '.join(system_names) or '(none configured)'}\n"
                    f"Allowed question types:\n{question_type_listing}\n\n"
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
                    "the question type names exactly as given -- most "
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
    unclear (or multi-type) classification there is left as whatever the
    classifier returns rather than blocking the turn on a second question.
    It's a list, not a single value, since a question can genuinely be more
    than one type -- see agents/question_types.py for the canonical values,
    enforced directly by _IntentClassification's Literal-typed field.
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
        "resolved_question_type": result.question_type,
    }


async def resolve_synonyms(state: DeterministicState) -> dict:
    """Ticket 21: resolves each extracted term against the canonical
    synonym graph directly, term by term -- a multi-word sentence
    essentially never matches a Term/CanonicalTerm node as a literal
    string, so the old whole-query call was a no-op on nearly every real
    turn despite the graph having the right data (confirmed directly
    against Neo4j in the 2026-09-03 eval reflection).

    Calls get_graph_client().get_synonym_groups() directly rather than
    going through the shared synonym_resolve tool (agents/tools/
    synonym_resolve.py, still used as-is by react_agent) -- that tool
    returns only the *first* matching group's other members for a term,
    silently discarding the fact that a term can match more than one
    distinct canonical group. This node needs to see every match to
    detect that.

    resolved_canonical_terms (2026-09-05) is the one canonical name per
    matched group, not a flat mix of canonical names and aliases -- every
    downstream consumer (hybrid_retrieve, resolve_skus) now retrieves/
    resolves against canonical wording specifically, since catalog
    descriptions are written that way ("GUIDEPIN...") and an alias adds
    nothing a canonical term doesn't already cover once matched.

    synonym_ambiguity flags a term that matched MORE than one distinct
    canonical group -- the rep's own word is genuinely ambiguous between
    two unrelated concepts (not just "the query mentions several different
    things", which is normal and not ambiguous) -- mapping that term to
    every canonical option it could mean. _should_clarify_synonyms routes
    straight to request_clarification when this is set, rather than
    silently picking one meaning.
    """
    search_query = state.get("search_query") or state["query"]
    terms = with_singular_variants(extract_candidate_terms(search_query))[:MAX_TERMS_PER_TURN]
    groups = await get_graph_client().get_synonym_groups()

    canonical_terms: dict[str, None] = {}
    ambiguity: dict[str, list[str]] = {}
    for term in terms:
        matches = [group for group in groups if any(variant.lower() == term for variant in group)]
        for group in matches:
            canonical_terms.setdefault(group[0], None)
        if len(matches) > 1:
            ambiguity[term] = [group[0] for group in matches]

    return {
        "resolved_canonical_terms": list(canonical_terms),
        "synonym_ambiguity": ambiguity or None,
    }


def _should_clarify_synonyms(
    state: DeterministicState,
) -> Literal["request_clarification", "hybrid_retrieve"]:
    if state.get("synonym_ambiguity") and state.get("clarification_rounds", 0) < 1:
        return "request_clarification"
    return "hybrid_retrieve"


# doctype-hierarchy.csv's priority order, expressed in this system's real
# document_type names (tags/repository.py's list_document_types()), not the
# CSV's own generic wording ("Inventory Control Form" here is "Inventory
# Control", etc.), keyed by agents/question_types.py's canonical slugs
# (2026-09-05, was the CSV's own verbose question-type wording). That CSV's
# own Priority 1 ("Master Item File") isn't a document at all -- it's the
# graph, already covered by resolve_skus/aggregate_facts (ticket 20) -- so
# this ranks what's left: the P2-P4 document-chunk tiers. Matched by
# substring, since a real document_type name and the CSV's wording don't
# line up exactly. Used twice: hard-filters hybrid_retrieve's retrieval
# pool to these types (plus anything untagged), and rerank's
# _doctype_priority_bonus below still ranks an allowed type above an
# untagged chunk within that pool -- filter narrows *which* documents are
# eligible, the bonus decides *ordering* among them.
#
# compatibility_lookup has no CSV-era precedent (canonicalized in from
# scratch, not carried over from an existing question type) -- given the
# same priority order as the two spec types is a reasonable default
# (compatibility facts are catalog/cross-reference-shaped, same as specs),
# not a verified-against-real-data choice the way the other three are.
_DOCTYPE_PRIORITY: dict[str, list[str]] = {
    "system_contents": ["Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique"],
    "product_characteristics": [
        "Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique",
    ],
    "compatibility_lookup": [
        "Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique",
    ],
    "technique_procedural": ["Surgical Technique", "Setup Guide", "Launch Presentation"],
}
# Additive score bonus per priority rank -- kept small relative to rerank's
# normalized-to-[0,1] BM25 score (see rerank below), so it breaks ties/
# favors a more-authoritative source on close calls rather than overriding
# a much more keyword-relevant passage from a lower-tier doctype.
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


def _allowed_document_type_ids(
    question_types: list[str], document_types: list[TagRecord]
) -> list[str] | None:
    """Resolves _DOCTYPE_PRIORITY's per-question-type allowed doctype names
    to real document_type ids, for hybrid_retrieve's hard filter. Returns
    None (no filter -- retrieve unfiltered) when none of this turn's
    resolved_question_type values have a defined priority order (e.g.
    "documents_lookup"), since restricting to an arbitrary/empty list would
    be worse than not filtering at all when there's no actual rule to apply.
    """
    allowed_names: set[str] = set()
    have_rule = False
    for question_type in question_types:
        order = _DOCTYPE_PRIORITY.get(question_type)
        if order:
            have_rule = True
            allowed_names.update(name.lower() for name in order)
    if not have_rule:
        return None
    return [
        str(dt.id) for dt in document_types if any(name in dt.name.lower() for name in allowed_names)
    ]


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
    # Canonical terms, not a flat canonical+alias mix -- see
    # resolve_synonyms' docstring.
    search_text = " ".join(filter(None, [search_query, *state.get("resolved_canonical_terms", [])]))
    question_types = state.get("resolved_question_type") or []
    # Hard-filters the retrieval pool by document type when the classified
    # question type has a defined preference (_DOCTYPE_PRIORITY) -- added
    # 2026-09-04 after comparing a Technique/procedural answer against its
    # golden expected answer found the real step-by-step Surgical Technique
    # narrative chunks ("Step 1. Surgical Approach", ...) never reaching the
    # retrieval pool at all: they ranked below inventory-table/marketing
    # chunks on raw semantic+keyword similarity, and rerank's doctype bonus
    # (a tiebreak among candidates already retrieved) can't promote a
    # passage that was never retrieved in the first place. Filtering here,
    # not just re-ranking after, is what actually keeps the wrong-shaped
    # chunks out of contention. Still admits untagged chunks regardless
    # (RetrievalFilters' own doc) -- this narrows the pool to relevant
    # *and* unknown-type documents, never to relevant-only.
    document_types = await list_document_types()
    allowed_document_type_ids = _allowed_document_type_ids(question_types, document_types)
    results = await vector_search.ainvoke(
        {
            "query": search_text,
            "top_k": RETRIEVE_TOP_K,
            "document_type_ids": allowed_document_type_ids,
        }
    )
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


_TOKEN_PATTERN = re.compile(r"[a-z0-9.]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _bm25_scores(query_terms: list[str], documents: list[str]) -> list[float]:
    """Standard Okapi BM25 over `documents` against `query_terms`'
    tokenized vocabulary -- no external dependency, this corpus (a dozen
    already-retrieved candidate passages) is far too small to need one.
    IDF is computed against just this candidate set, not the full corpus
    (there's no cheap access to global document-frequency stats here, and
    re-ranking only ever needs *relative* ordering among these candidates
    anyway, which a local IDF still gives correctly).
    """
    tokenized_docs = [_tokenize(doc) for doc in documents]
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    n_docs = len(tokenized_docs)

    query_tokens = list(dict.fromkeys(_tokenize(" ".join(query_terms))))
    doc_freq = {token: sum(1 for doc in tokenized_docs if token in doc) for token in query_tokens}

    scores = []
    for doc, length in zip(tokenized_docs, doc_lengths):
        score = 0.0
        for token in query_tokens:
            n_t = doc_freq[token]
            if n_t == 0:
                continue
            idf = math.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1)
            tf = doc.count(token)
            norm_len = length / avg_len if avg_len else 0.0
            denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * norm_len)
            score += idf * (tf * (_BM25_K1 + 1)) / denom if denom else 0.0
        scores.append(score)
    return scores


def _normalize_to_unit_range(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        # Every candidate scored identically (including all-zero, the
        # common case for a query with no term overlap at all) -- leave
        # the doctype bonus as the sole differentiator rather than
        # dividing by zero.
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


async def rerank(state: DeterministicState) -> dict:
    """Ranks hybrid_retrieve's candidate pool by BM25 keyword relevance
    plus the doctype-priority bonus -- no LLM call (2026-09-05, replacing
    an LLM relevance-scoring pass): rerank's job is picking among
    candidates hybrid_retrieve already retrieved as plausibly relevant
    (RRF-fused vector+full-text search), not open-ended judgment, so a
    classic keyword-overlap ranking over that already-narrowed pool is
    cheaper and removes a full model round-trip from every turn.

    Scored against the query plus this turn's resolved canonical terms
    (agents/question_types.py-classified retrieval already uses these to
    build the search text in hybrid_retrieve; scoring with the same
    vocabulary keeps rerank consistent with what was actually searched
    for, not just the rep's raw wording).

    BM25 scores are normalized to [0, 1] before adding the doctype bonus --
    raw BM25 has no fixed scale (unlike the old LLM pass's native 0-1
    output), so leaving it unnormalized would make _DOCTYPE_BONUS_STEP's
    calibration (tuned to matter at 0-1 scale) meaningless noise by
    comparison on some turns and dominant on others.
    """
    passages = state["retrieved"]
    if not passages:
        return {"reranked": []}
    query_terms = [state["query"], *state.get("resolved_canonical_terms", [])]
    documents = [passage["text"] for passage in passages]
    bm25 = _normalize_to_unit_range(_bm25_scores(query_terms, documents))
    question_types = state.get("resolved_question_type") or []
    # score is overwritten with the combined (BM25 + doctype priority
    # bonus) value, not left at hybrid_retrieve's original RRF fusion
    # score -- that's the number that actually decided this passage's
    # rank, and a trace/debugging view showing the old, pre-rerank score
    # here would misrepresent why a given passage made the cut.
    combined = [
        {**passage, "score": score + _doctype_priority_bonus(question_types, passage.get("document_type"))}
        for passage, score in zip(passages, bm25)
    ]
    scored = sorted(combined, key=lambda passage: passage["score"], reverse=True)
    return {"reranked": scored[:RERANK_TOP_N]}


async def resolve_skus(state: DeterministicState) -> dict:
    """Ticket 22: resolves this turn to candidate Part nodes so generate
    can ground spec/SKU/pairing facts in the graph (ticket 20) instead of
    vector prose alone. Always resolves the rep's own query terms against
    the graph; "technique_procedural" questions additionally resolve terms
    from the *retrieved* passage text (e.g. an inventory-table chunk naming
    exact SKUs next to the step that uses them) -- found in the 2026-09-03
    eval reflection that query-alone wasn't enough there: a procedural
    question can name specifics (a screw size, a thread-type word) the
    reranked top-5 doesn't happen to surface a matching SKU chunk for, so
    resolve_skus had nothing to resolve regardless of how good the
    resolution mechanism itself was -- e.g. "a 4.0 and 3.5 screw for their
    bunion" naming exactly the sizes needed, with the Full-Thread inventory
    chunk not making that turn's top-5.

    resolved_canonical_terms (already computed this turn by
    resolve_synonyms, above) is folded in for free -- e.g. a query that
    says "wire" already resolved "guidepin" as its canonical term, which is
    the term that actually substring-matches Part descriptions.

    Multiple candidate Parts for one term are never narrowed to one here --
    all matches are deduped by SKU and handed to generate labeled, which
    decides what the question actually needs. Disambiguating up front (or
    interrupt()-ing like detect_intent does for system ambiguity) would
    fire on completely ordinary questions, since one query legitimately
    matching several real SKUs is the common case, not the exception. (A
    genuinely ambiguous *word*, as opposed to an ordinary multi-match
    lookup, is instead caught earlier by resolve_synonyms/
    _should_clarify_synonyms -- see there.)

    2026-09-04: does its own query-alone lookup again (removed ticket 24's
    resolve_query_entities node, which used to compute this once upfront
    for both hybrid_retrieve's search-text enrichment and this node to
    reuse) -- that enrichment was found to actively hurt technique_
    procedural retrieval (see hybrid_retrieve's docstring), and this node
    needs its own graph lookup regardless of whether an earlier node ever
    ran one, so folding both concerns into one upfront node was no longer
    worth the coupling. Costs one extra round of Neo4j lookups per turn
    versus the ticket-24 shape; still per-term, still cheap.
    """
    question_types = state.get("resolved_question_type") or []
    family = state.get("resolved_system")

    query_terms = with_singular_variants(
        [*extract_candidate_terms(state["query"]), *state.get("resolved_canonical_terms", [])]
    )
    all_terms = query_terms
    if "technique_procedural" in question_types:
        passage_text = " ".join(passage["text"] for passage in (state.get("reranked") or []))
        all_terms = with_singular_variants([*query_terms, *extract_candidate_terms(passage_text)])
    all_terms = all_terms[:MAX_TERMS_PER_TURN]

    results = await asyncio.gather(
        *(part_lookup.ainvoke({"term": term, "product_family": family}) for term in all_terms)
    )
    matched: dict[str, dict] = {}
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
# turn as (at least partly) "technique_procedural" -- a spec-lookup answer
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
    # clarification round, generate runs again with an enriched search
    # (see request_clarification), and only the version self_eval accepts
    # should become permanent conversation history -- see finalize below,
    # not this node, for the actual `messages` append.
    history = state["messages"][:-1]
    clarification_reply = state.get("clarification_reply")
    clarification_note = (
        f"\n\nThe rep clarified: {clarification_reply}" if clarification_reply else ""
    )
    augmented_question = HumanMessage(
        content=(
            f"Known catalog facts:\n{_format_known_facts(state.get('aggregated_facts', ''))}"
            f"\n\nContext:\n{_format_context(passages)}\n\nQuestion: {state['query']}"
            f"{clarification_note}"
        )
    )
    system_prompt = _GENERATE_SYSTEM_PROMPT
    if "technique_procedural" in (state.get("resolved_question_type") or []):
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


class _ClarifyingQuestion(BaseModel):
    question: str = Field(
        description=(
            "One short, specific question to ask the sales rep that would "
            "supply whatever information is missing or ambiguous."
        )
    )


async def _generate_clarifying_question(query: str, draft_answer: str) -> str:
    model = get_chat_model().with_structured_output(_ClarifyingQuestion)
    result = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "The draft answer below wasn't well-supported by the "
                    "available product documentation/catalog data. Write one "
                    "short, specific question to ask the sales rep that would "
                    "supply whatever's missing or ambiguous -- e.g. naming a "
                    "system, procedure, or part detail the original question "
                    "left unclear. Don't mention scoring, confidence, or "
                    "documentation; just ask the question directly, the way "
                    "a knowledgeable colleague would."
                )
            ),
            {"role": "user", "content": f"Rep's question: {query}\n\nDraft answer: {draft_answer}"},
        ]
    )
    return result.question


def _synonym_ambiguity_question(ambiguity: dict[str, list[str]]) -> str:
    """Builds the clarifying question directly from what resolve_synonyms
    already found -- no LLM call needed, unlike the self_eval-triggered
    path below: we already know exactly which word is ambiguous and
    exactly what it could mean, so asking a model to guess a question would
    only add latency and a chance of asking something vaguer than this.
    """
    # No .capitalize() -- it lowercases everything *after* the first
    # letter too, which would mangle a canonical term with real uppercase
    # in it (a SKU-like name, say); the leading "Just" is already
    # capitalized as written below.
    clauses = [f'by "{term}" do you mean {" or ".join(options)}' for term, options in ambiguity.items()]
    return "Just to confirm, " + "; and ".join(clauses) + "?"


async def request_clarification(state: DeterministicState) -> dict:
    """Replaces an earlier silent reformulate-and-retry loop (removed
    2026-09-04): a weak self_eval score used to trigger an LLM-guessed
    query rewrite with no new information, which could still miss and ship
    a low-confidence answer as though it were settled -- and, per the
    2026-09-03/04 eval reflections, the extra pass-per-retry pushed a full
    2-retry run's node count past LangGraph's default recursion_limit.

    Now the graph pauses via the same interrupt()/resume mechanic
    detect_intent uses (ticket 09) and asks the rep directly, exactly once
    per turn (clarification_rounds caps it -- see _should_clarify /
    _should_clarify_synonyms, both routing here) rather than the graph
    guessing: their reply is real new information, not a model's guess,
    and it's threaded into both search_query (so the retry's retrieval can
    use it) and clarification_reply (so generate's prompt can reason over
    it directly, not just hope better retrieval surfaces it).

    Two distinct callers, two distinct question sources (2026-09-05):
    resolve_synonyms/_should_clarify_synonyms routes here BEFORE retrieval
    or generate have run at all (synonym_ambiguity is set, state has no
    answer yet) -- self_eval/_should_clarify routes here AFTER a full pass
    produced a weak draft answer (synonym_ambiguity is absent, state.answer
    is the thing to react to). Checking which is set is what picks the
    right question source below; a state carrying neither at once would be
    a graph-wiring bug, not a real turn shape.

    Like detect_intent's own interrupt() call, everything above the
    interrupt() line below re-runs on resume (LangGraph's interrupt/resume
    contract replays the node from its start) -- an extra clarifying-
    question LLM call on resume for the self_eval path, same acceptable
    cost detect_intent already pays for its own classification call (the
    synonym-ambiguity path has no LLM call to repeat either way).
    """
    ambiguity = state.get("synonym_ambiguity")
    if ambiguity:
        question = _synonym_ambiguity_question(ambiguity)
    else:
        question = await _generate_clarifying_question(state["query"], state.get("answer", ""))
    reply = interrupt({"question": question, "options": []})
    search_query = f"{state.get('search_query') or state['query']} {reply}"
    return {
        "search_query": search_query,
        "clarification_reply": reply,
        "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        "synonym_ambiguity": None,
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


def _should_clarify(state: DeterministicState) -> Literal["request_clarification", "finalize"]:
    scores = state.get("eval_scores") or {}
    rounds = state.get("clarification_rounds", 0)
    scored_low = (
        scores.get("faithfulness", 1.0) < CLARIFICATION_SCORE_THRESHOLD
        or scores.get("relevance", 1.0) < CLARIFICATION_SCORE_THRESHOLD
    )
    if scored_low and rounds < 1:
        return "request_clarification"
    return "finalize"


def build_graph(checkpointer):
    graph = StateGraph(DeterministicState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("resolve_synonyms", resolve_synonyms)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("resolve_skus", resolve_skus)
    graph.add_node("aggregate_facts", aggregate_facts)
    graph.add_node("generate", generate)
    graph.add_node("self_eval", self_eval)
    graph.add_node("request_clarification", request_clarification)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "resolve_synonyms")
    graph.add_conditional_edges("resolve_synonyms", _should_clarify_synonyms)
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "resolve_skus")
    graph.add_edge("resolve_skus", "aggregate_facts")
    graph.add_edge("aggregate_facts", "generate")
    graph.add_edge("generate", "self_eval")
    graph.add_conditional_edges("self_eval", _should_clarify)
    graph.add_edge("request_clarification", "resolve_synonyms")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


register("deterministic", build_graph)

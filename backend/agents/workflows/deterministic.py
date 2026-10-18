"""Deterministic retrieval workflow: fixed pipeline, no agentic tool choice.

query -> detect_intent -> resolve_synonyms
       -> (if a term is genuinely ambiguous between canonical concepts:
           request_clarification, then back to resolve_synonyms)
       -> hybrid_retrieve -> rerank -> resolve_skus -> aggregate_facts
       -> generate -> self_eval
       -> (if the fact check found problems and no correction pass has
           run yet this turn: generate again with the problems listed,
           then self_eval again)
       -> finalize

detect_intent classifies which Product Family the query is about and,
  when it can't tell confidently, calls interrupt() to ask -- offering the
  classifier's few candidate families (or every family, if it had none)
  plus free text. hybrid_retrieve then hard-filters to that family.

resolve_synonyms/_should_clarify_synonyms: when a single
  extracted word matches more than one distinct canonical concept (the
  rep's own word is ambiguous, not just "the query mentions several
  things"), the graph asks which one they mean before ever running
  retrieval on a guess.

self_eval is a deterministic fact check (agents/fact_check.py) with
no model call -- a failed check means the model misstated something the
sources already settle (a wrong thread type, an invented SKU, a citation
to a passage it never saw), which the rep can't clarify, so it earns one
correction pass with the specific problems listed instead of a question.

hybrid_retrieve also hard-filters its retrieval pool by document type
based on the classified question type (the admin-edited question type ->
doc type hierarchy table, doc_type_hierarchy/, matched by doc type id --
issue #17), rather than only
softly re-ranking afterward -- found live comparing an answer against its
golden expected answer that a technique_procedural question's real
step-by-step narrative chunks were being crowded out of the retrieval pool
entirely by inventory-table/marketing chunks that happened to rank higher
on raw semantic+keyword similarity; filtering the *retrieval* step itself,
not just the rerank tiebreak, is what actually keeps them out of
contention. The doc-type filter always still admits chunks with no doc
type; the family filter (issue #18) does not -- see RetrievalFilters'
docstring.

rerank scores that retrieval pool with BM25 (a classic keyword-overlap
ranking replacing an LLM relevance-scoring pass -- see its own
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
from agents.fact_check import check_answer, format_issues_for_correction
from agents.question_types import QUESTION_TYPE_NAMES, QUESTION_TYPES
from agents.registry import register
from agents.state import BaseAgentState, RetrievedPassage
from agents.tools.part_lookup import part_lookup
from agents.tools.term_extraction import extract_candidate_terms, with_singular_variants
from agents.tools.vector_search import vector_search
from catalog.display_names import CatalogDisplayNames, catalog_display_names_for
from catalog.repository import ProductFamilyRecord, list_families_with_trays
from config.llm_clients import get_chat_model
from doc_type_hierarchy.repository import RankedDocumentType, list_hierarchy
from retrieval.graph_client import get_graph_client

# Neo4j round-trips are cheap individually but resolve_skus can extract
# dozens of candidate terms from a long procedural passage -- capped so one
# turn can't fire an unbounded number of concurrent graph queries.
MAX_TERMS_PER_TURN = 40

logger = logging.getLogger(__name__)

# How many times per turn a draft that failed self_eval's fact check gets
# regenerated with its problems listed -- see _should_correct. One pass
# fixes what it can; a draft still failing after that ships as-is rather
# than looping.
MAX_CORRECTION_ROUNDS = 1
# hybrid_retrieve pulls a wider candidate pool than actually goes to
# generation -- rerank (a BM25 keyword-relevance pass over that pool, plus
# a doctype-priority bonus -- no LLM call, see rerank's own docstring) is
# what narrows it down, so it needs real candidates to choose among.
RETRIEVE_TOP_K = 12
RERANK_TOP_N = 5
# How many Product Families detect_intent's clarification offers when the
# classifier is unsure (issue #18) -- all 18 at once is too many to scan.
MAX_FAMILY_CANDIDATES = 4

# BM25's standard tuning constants (Okapi BM25) -- k1 controls term-frequency
# saturation, b controls document-length normalization strength. 1.5/0.75
# are the conventional defaults, not tuned against this corpus specifically.
_BM25_K1 = 1.5
_BM25_B = 0.75


class DeterministicState(BaseAgentState, total=False):
    # `query` (from BaseAgentState) is always the user's actual question,
    # verbatim -- generate echoes it back, so nothing past detect_intent may touch it. `search_query` is
    # what retrieval actually uses; it starts equal to `query` and is the
    # only thing request_clarification rewrites (appending the rep's reply).
    # Both search_query and the clarification fields below are reset per
    # turn by the API layer (backend/api/routes/chat.py) -- see its inputs
    # dict -- since none has a reducer and would otherwise leak a prior
    # turn's leftover value in via the checkpointer.
    search_query: str
    # Set by resolve_synonyms --
    # see that node's own docstring for why canonical terms specifically,
    # not a flat canonical+alias mix. synonym_ambiguity is only ever
    # non-None for the one turn between resolve_synonyms detecting a
    # genuinely ambiguous term and request_clarification resolving it (see
    # _should_clarify_synonyms) -- not meant to persist as a fact about the
    # turn the way resolved_canonical_terms does.
    resolved_canonical_terms: list[str]
    synonym_ambiguity: dict[str, list[str]] | None
    reranked: list[RetrievedPassage]
    # Set by resolve_skus/aggregate_facts -- resolved_parts is
    # every candidate Part node matched from this turn's extracted terms,
    # deduped by SKU; aggregated_facts is that same data formatted into
    # generate's context. Multiple candidates are never disambiguated down
    # to one here -- see resolve_skus's docstring for why.
    resolved_parts: list[dict]
    aggregated_facts: str
    answer: str
    citations: list[str]
    # Set by request_clarification when resolve_synonyms finds an ambiguous
    # term -- clarification_reply is the rep's answer to the question it
    # asked (folded into generate's prompt so the model can actually use
    # it, not just into search_query for retrieval); clarification_rounds
    # caps this to once per turn -- see _should_clarify_synonyms.
    clarification_reply: str | None
    clarification_rounds: int
    # How many fact-check correction passes generate has run this turn --
    # see _should_correct. Reset per turn by the API layer like the
    # clarification fields above.
    correction_rounds: int
    # Set by detect_intent, the graph's entry node (issue #18).
    # resolved_family_id is what hybrid_retrieve hard-filters on;
    # resolved_family is that same family's display name, kept alongside it
    # for resolve_skus' graph scoping (part_lookup maps a display name back
    # to the catalog name) and react_agent's prompt. Both are None together
    # when no family was resolved -- nothing configured, or a clarification
    # reply that names no real family -- and retrieval then runs unscoped.
    # resolved_question_type drives hybrid_retrieve's document-type filter
    # (the doc type hierarchy table, issue #17) in addition to its original
    # observability role. It's a list, not a single value, since a question
    # can genuinely span more than one type (e.g. both a procedural step
    # and a product spec in the same question) -- detect_intent does NOT
    # interrupt on a multi-type classification, only on family ambiguity
    # (see its own docstring). Always one of agents/question_types.py's
    # QUESTION_TYPE_NAMES, enforced by _IntentClassification's Literal
    # field -- not a free-form string.
    resolved_family: str | None
    resolved_family_id: str | None
    resolved_question_type: list[str]
    # Set by hybrid_retrieve from the doc type hierarchy table: document
    # type id -> rerank's additive priority bonus for this turn's question
    # types. Carried to rerank in state so the table is read once per turn.
    document_type_bonuses: dict[str, float]


class _IntentClassification(BaseModel):
    # Free strings checked against the real families afterwards
    # (_resolve_family), not a Literal enum like question_type: families
    # are data (admin-renamable display names), and anything that isn't a
    # real family -- including the literal "null" structured output
    # sometimes writes instead of leaving the field unset -- simply
    # resolves to no family.
    product_family: str | None = Field(
        default=None,
        description=(
            "The one product family name (from the allowed list given in the "
            "prompt) the query is clearly about, or null if it doesn't "
            "clearly name or imply exactly one -- don't guess between two "
            "plausible families."
        ),
    )
    candidate_families: list[str] = Field(
        default_factory=list,
        description=(
            f"Only when product_family is null: up to {MAX_FAMILY_CANDIDATES} "
            "allowed product family names the query could plausibly be "
            "about, most likely first. Empty if none is plausible."
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
            "the same question). Use unclassified, alone, if none of the "
            "others clearly applies."
        ),
    )


def _resolve_family(
    name: str | None, families: list[ProductFamilyRecord], display_names: CatalogDisplayNames
) -> ProductFamilyRecord | None:
    """Turns a family name -- the classifier's pick, a clicked clarification
    option, or a rep's free-text reply -- back into the real family it
    names: its display name or catalog name, case-insensitively. None for
    anything else, so a reply naming no family leaves the turn unscoped
    rather than recording a family that doesn't exist.
    """
    if not name or not name.strip():
        return None
    catalog_name = display_names.family_catalog_name(name.strip()).lower()
    return next((family for family in families if family.name.lower() == catalog_name), None)


def _clarification_options(
    candidate_names: list[str], families: list[ProductFamilyRecord], display_names: CatalogDisplayNames
) -> list[str]:
    """The family display names a clarification offers: the classifier's
    candidates that are real families, deduped and capped at
    MAX_FAMILY_CANDIDATES -- or every family when none of them is.
    """
    options: dict[str, None] = {}
    for name in candidate_names:
        family = _resolve_family(name, families, display_names)
        if family is not None:
            options.setdefault(family.display_name, None)
    return list(options)[:MAX_FAMILY_CANDIDATES] or [family.display_name for family in families]


async def _classify_intent(query: str, family_names: list[str]) -> _IntentClassification:
    model = get_chat_model().with_structured_output(_IntentClassification)
    question_type_listing = "\n".join(f"- {question_type.name}: {question_type.description}" for question_type in QUESTION_TYPES)
    return await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Classify the sales rep's question below.\n\n"
                    f"Allowed product families: {', '.join(family_names) or '(none configured)'}\n"
                    f"Allowed question types:\n{question_type_listing}\n\n"
                    "Set product_family to null if the question doesn't "
                    "clearly name or imply exactly one of the allowed "
                    "product families -- when genuinely unclear, always "
                    "prefer null over guessing, even if only one family is "
                    "configured. When it's null, list in candidate_families "
                    f"the (at most {MAX_FAMILY_CANDIDATES}) allowed families "
                    "the question could plausibly be about, most likely "
                    "first, or none if nothing in it points anywhere. Only "
                    "once the question does clearly identify a family: if "
                    "several allowed families share a common brand name "
                    "(e.g. MIS or REFLEX), pick the one that actually "
                    "matches, and don't switch to a different family just "
                    "because of incidental word overlap (a question about a "
                    "screw, wire, or guidepin describes the implant/"
                    "consumable family, not a separate powered-instrument "
                    "family, unless it explicitly asks about the powered "
                    "tool itself). Use family names and question type names "
                    "exactly as given -- most questions are exactly one "
                    "type; if a question can be interpreted as more than one "
                    "type, output multiple types."
                )
            ),
            {"role": "user", "content": query},
        ]
    )


async def detect_intent(state: DeterministicState) -> dict:
    """Entry node: classifies which Product Family and question type the
    query is about (issue #18). A query that doesn't clearly name or imply
    one family pauses the graph via interrupt() for a clarifying answer,
    rather than letting hybrid_retrieve search across every family's
    documents -- backend/api/routes/chat.py's resume_chat is the other half
    of this suspend/resume pair, and _resolve_family is what turns either a
    clicked option or free text back into a real family on resume.

    The question offers the classifier's candidate families (at most
    MAX_FAMILY_CANDIDATES -- showing all 18 is too many), or every family
    when it had no real candidates; free text may name any family, not
    just an offered one. At most one family clarification per turn: a
    reply that names no real family leaves the turn unscoped (retrieval
    unfiltered by family) rather than asking again.

    Families come from the Postgres catalog mirror (catalog.repository),
    by display name, so an admin rename shows up here with no code change.
    If none are mirrored yet (a fresh environment before the catalog
    seed), there's nothing to disambiguate against, so this skips straight
    past the interrupt and leaves resolved_family None.

    question_type never triggers this interrupt -- an unclear question
    is classified `unclassified`, which simply retrieves unfiltered (see
    the doc type hierarchy notes above hybrid_retrieve), so an unclear (or
    multi-type) classification there is left as whatever the classifier
    returns rather than blocking the turn on a second question.
    It's a list, not a single value, since a question can genuinely be more
    than one type -- see agents/question_types.py for the canonical values,
    enforced directly by _IntentClassification's Literal-typed field.
    """
    families = await list_families_with_trays()
    display_names = catalog_display_names_for(families)
    if not families:
        # Expected only before the master catalog seed has run. Logged, not
        # raised: skipping the interrupt and retrieving unscoped is still
        # the right behavior -- this just makes the case observable.
        logger.warning(
            "detect_intent: no product families mirrored for query=%r "
            "-- family clarification skipped this turn",
            state["query"],
        )

    result = await _classify_intent(state["query"], [family.display_name for family in families])
    family = _resolve_family(result.product_family, families, display_names)

    if family is None and families:
        # Everything above re-runs on resume (LangGraph replays the node
        # from its start), the classifier call included.
        answer = interrupt(
            {
                "question": "Which product family is this about?",
                "options": _clarification_options(result.candidate_families, families, display_names),
            }
        )
        family = _resolve_family(answer, families, display_names)

    return {
        "resolved_family": family.display_name if family else None,
        "resolved_family_id": str(family.id) if family else None,
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


# Which document types each question type retrieves from, ranked, is data
# (issue #17, docs/adr/0006-doc-type-hierarchy-is-data.md): the
# question_type_document_types table, edited from the Settings page and
# read once per turn by hybrid_retrieve (doc_type_hierarchy.repository's
# list_hierarchy). It replaced an in-code _DOCTYPE_PRIORITY dict of name
# substrings -- the table is matched by document type id, so renaming a doc
# type can't silently drop it out of retrieval, and a new one is wired in
# without a deploy. That dict's contents seeded the table (see the
# ..._create_question_type_document_types.py migration).
#
# Used twice: hard-filters hybrid_retrieve's retrieval pool to the mapped
# types (plus anything untagged), and rerank's bonus still ranks an allowed
# type above an untagged chunk within that pool -- filter narrows *which*
# documents are eligible, the bonus decides *ordering* among them. A
# question type with nothing mapped -- always including `unclassified`,
# and e.g. `documents_lookup` -- applies no filter.
#
# Additive score bonus per priority rank -- kept small relative to rerank's
# normalized-to-[0,1] BM25 score (see rerank below), so it breaks ties/
# favors a more-authoritative source on close calls rather than overriding
# a much more keyword-relevant passage from a lower-tier doctype.
_DOCTYPE_BONUS_STEP = 0.05


def _ranked_document_type_ids(
    hierarchy: dict[str, list[RankedDocumentType]],
) -> dict[str, list[str]]:
    return {
        question_type: [str(document_type.id) for document_type in document_types]
        for question_type, document_types in hierarchy.items()
    }


def _document_type_bonuses(
    question_types: list[str], ranking: dict[str, list[str]]
) -> dict[str, float]:
    """Document type id -> rerank bonus for this turn. The first of this
    turn's question types with anything mapped decides the order (most
    authoritative gets the biggest bonus); a doc type it doesn't rank gets
    none. Empty when no question type has anything mapped.
    """
    for question_type in question_types:
        ranked_ids = ranking.get(question_type)
        if ranked_ids:
            return {
                document_type_id: (len(ranked_ids) - rank) * _DOCTYPE_BONUS_STEP
                for rank, document_type_id in enumerate(ranked_ids)
            }
    return {}


def _allowed_document_type_ids(
    question_types: list[str], ranking: dict[str, list[str]]
) -> list[str] | None:
    """The document type ids hybrid_retrieve hard-filters to: every id
    mapped to any of this turn's question types. Returns None (no filter --
    retrieve unfiltered) when none of them has anything mapped (e.g.
    "unclassified" or "documents_lookup"), since restricting to an empty
    list would be worse than not filtering at all when there's no actual
    rule to apply.
    """
    allowed_ids: dict[str, None] = {}
    for question_type in question_types:
        for document_type_id in ranking.get(question_type, []):
            allowed_ids.setdefault(document_type_id, None)
    return list(allowed_ids) or None


async def hybrid_retrieve(state: DeterministicState) -> dict:
    # Hard-filtered by detect_intent's resolved family (issue #18): a
    # family-scoped turn only sees chunks of documents tagged with that
    # family, and a document with no family is excluded -- accepted, since
    # the legacy-tag migration (catalog/migrate_legacy_system_tags.py) is
    # what gives existing documents their family. This replaced a
    # deliberate "not filtered by system" rule that existed only because
    # the old system tag was optional. No family resolved -> no family
    # filter.
    search_query = state.get("search_query") or state["query"]
    # Canonical terms, not a flat canonical+alias mix -- see
    # resolve_synonyms' docstring.
    search_text = " ".join(filter(None, [search_query, *state.get("resolved_canonical_terms", [])]))
    question_types = state.get("resolved_question_type") or []
    # Hard-filters the retrieval pool by document type when the classified
    # question type has doc types mapped (the hierarchy table) -- added
    # 2026-09-04 after comparing a Technique/procedural answer against its
    # golden expected answer found the real step-by-step Surgical Technique
    # narrative chunks ("Step 1. Surgical Approach", ...) never reaching the
    # retrieval pool at all: they ranked below inventory-table/marketing
    # chunks on raw semantic+keyword similarity, and rerank's doctype bonus
    # (a tiebreak among candidates already retrieved) can't promote a
    # passage that was never retrieved in the first place. Filtering here,
    # not just re-ranking after, is what actually keeps the wrong-shaped
    # chunks out of contention. Still admits chunks with no doc type
    # regardless (RetrievalFilters' own doc) -- this narrows the pool to
    # relevant *and* unknown-type documents, never to relevant-only.
    ranking = _ranked_document_type_ids(await list_hierarchy())
    allowed_document_type_ids = _allowed_document_type_ids(question_types, ranking)
    results = await vector_search.ainvoke(
        {
            "query": search_text,
            "top_k": RETRIEVE_TOP_K,
            "document_type_ids": allowed_document_type_ids,
            "product_family_id": state.get("resolved_family_id"),
        }
    )
    retrieved: list[RetrievedPassage] = [
        {
            "chunk_id": result["citation"],
            "document_id": result["document_id"],
            "text": result["content"],
            "score": result["score"],
            "document_type": result.get("document_type"),
            "document_type_id": result.get("document_type_id"),
        }
        for result in results
    ]
    return {
        "retrieved": retrieved,
        "document_type_bonuses": _document_type_bonuses(question_types, ranking),
    }


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
    tokenized_documents = [_tokenize(document) for document in documents]
    document_lengths = [len(document) for document in tokenized_documents]
    average_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
    document_count = len(tokenized_documents)

    query_tokens = list(dict.fromkeys(_tokenize(" ".join(query_terms))))
    document_frequency = {token: sum(1 for document in tokenized_documents if token in document) for token in query_tokens}

    scores = []
    for document, length in zip(tokenized_documents, document_lengths):
        score = 0.0
        for token in query_tokens:
            documents_with_term = document_frequency[token]
            if documents_with_term == 0:
                continue
            idf = math.log((document_count - documents_with_term + 0.5) / (documents_with_term + 0.5) + 1)
            term_frequency = document.count(token)
            normalized_length = length / average_length if average_length else 0.0
            denom = term_frequency + _BM25_K1 * (1 - _BM25_B + _BM25_B * normalized_length)
            score += idf * (term_frequency * (_BM25_K1 + 1)) / denom if denom else 0.0
        scores.append(score)
    return scores


def _normalize_to_unit_range(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lowest, highest = min(scores), max(scores)
    if highest == lowest:
        # Every candidate scored identically (including all-zero, the
        # common case for a query with no term overlap at all) -- leave
        # the doctype bonus as the sole differentiator rather than
        # dividing by zero.
        return [0.0 for _ in scores]
    return [(score - lowest) / (highest - lowest) for score in scores]


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
    # Doc type id -> bonus, computed by hybrid_retrieve from the hierarchy
    # table for this turn's question types. An untagged chunk (no id) or a
    # type the leading question type doesn't rank gets no bonus.
    document_type_bonuses = state.get("document_type_bonuses") or {}
    # score is overwritten with the combined (BM25 + doctype priority
    # bonus) value, not left at hybrid_retrieve's original RRF fusion
    # score -- that's the number that actually decided this passage's
    # rank, and a trace/debugging view showing the old, pre-rerank score
    # here would misrepresent why a given passage made the cut.
    combined = [
        {**passage, "score": score + document_type_bonuses.get(passage.get("document_type_id") or "", 0.0)}
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
    interrupt()-ing like detect_intent does for family ambiguity) would
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
    # A display name -- part_lookup maps it back to the catalog name.
    family = state.get("resolved_family")

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
    # (see request_clarification), and on a correction pass it runs again
    # with self_eval's fact-check issues -- only the final version should
    # become permanent conversation history -- see finalize below, not
    # this node, for the actual `messages` append.
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
    prompt = [SystemMessage(content=system_prompt), *history, augmented_question]
    # A correction pass (routed here by _should_correct): show the model
    # its own failed draft and exactly what the fact check found, rather
    # than re-asking from scratch and hoping it lands differently.
    issues = state.get("fact_check_issues") or []
    update: dict = {}
    if issues:
        prompt += [AIMessage(content=state["answer"]), HumanMessage(content=format_issues_for_correction(issues))]
        update["correction_rounds"] = state.get("correction_rounds", 0) + 1
    response = await model.ainvoke(prompt)
    answer = response.text
    return {**update, "answer": answer, "citations": extract_citations(answer)}


async def self_eval(state: DeterministicState) -> dict:
    """Fact-checks the draft (agents/fact_check.py) against everything
    generate was actually allowed to use -- the reranked passages *and* the
    resolved catalog Part records. Both matter for the same reason they
    did when this was an LLM judge (ticket 20/21/22's eval run): a claim
    correctly grounded in the catalog but absent from every passage is the
    graph-grounding ticket 20 exists for, not an unsupported claim.
    """
    passages = state.get("reranked") or []
    issues = check_answer(
        state["answer"],
        citation_ids={passage["chunk_id"] for passage in passages},
        source_text="\n".join(
            [*(passage["text"] for passage in passages), state.get("aggregated_facts") or ""]
        ),
        parts=state.get("resolved_parts") or [],
    )
    return {"fact_check_issues": issues}


def _synonym_ambiguity_question(ambiguity: dict[str, list[str]]) -> str:
    """Builds the clarifying question directly from what resolve_synonyms
    already found -- no LLM call needed: we already know exactly which word
    is ambiguous and exactly what it could mean, so asking a model to guess
    a question would only add latency and a chance of asking something
    vaguer than this.
    """
    # No .capitalize() -- it lowercases everything *after* the first
    # letter too, which would mangle a canonical term with real uppercase
    # in it (a SKU-like name, say); the leading "Just" is already
    # capitalized as written below.
    clauses = [f'by "{term}" do you mean {" or ".join(options)}' for term, options in ambiguity.items()]
    return "Just to confirm, " + "; and ".join(clauses) + "?"


async def request_clarification(state: DeterministicState) -> dict:
    """Pauses via the same interrupt()/resume mechanic detect_intent uses
    (ticket 09) to ask the rep which meaning of an ambiguous term they
    meant, exactly once per turn (clarification_rounds caps it -- see
    _should_clarify_synonyms) rather than the graph guessing: their reply
    is real new information, not a model's guess, and it's threaded into
    both search_query (so retrieval can use it) and clarification_reply (so
    generate's prompt can reason over it directly, not just hope better
    retrieval surfaces it).

    resolve_synonyms/_should_clarify_synonyms is the only caller, and it
    routes here BEFORE retrieval or generate have run at all. (self_eval
    also routed here until it became a deterministic fact check -- see the
    module docstring for why a failed check now earns a correction pass
    instead of a question.)

    Like detect_intent's own interrupt() call, everything above the
    interrupt() line below re-runs on resume (LangGraph's interrupt/resume
    contract replays the node from its start) -- harmless here, since
    building the question is a pure string format.
    """
    question = _synonym_ambiguity_question(state.get("synonym_ambiguity") or {})
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
    earlier attempt (a pre-clarification or pre-correction draft) never
    ends up alongside the final one -- only the answer actually shipped
    should shape future turns.

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


def _should_correct(state: DeterministicState) -> Literal["generate", "finalize"]:
    if state.get("fact_check_issues") and state.get("correction_rounds", 0) < MAX_CORRECTION_ROUNDS:
        return "generate"
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
    graph.add_conditional_edges("self_eval", _should_correct)
    graph.add_edge("request_clarification", "resolve_synonyms")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


register("deterministic", build_graph)

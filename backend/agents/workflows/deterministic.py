"""Deterministic retrieval workflow: fixed pipeline, no agentic tool choice.

query -> detect_intent -> clarify_question_type
       -> (documents_lookup only: lookup_document, which answers with the
           one document clearly asked for and goes straight to finalize --
           or, with no clear match, carries on below)
       -> resolve_synonyms
       -> (if a term is genuinely ambiguous between canonical concepts:
           request_clarification, then back to resolve_synonyms)
       -> hybrid_retrieve -> resolve_query_parts -> rerank -> resolve_skus
       -> aggregate_facts -> generate -> self_eval
       -> (if the fact check found problems and no correction pass has
           run yet this turn: generate again with the problems listed,
           then self_eval again)
       -> finalize

detect_intent classifies which Product Family the query is about and,
  when it can't tell confidently, calls interrupt() to ask -- offering the
  classifier's few candidate families (or every family, if it had none)
  plus free text. hybrid_retrieve then hard-filters to that family. A
  follow-up naming no family stays on the thread's previous one instead of
  asking again (issue #22).

clarify_question_type (issue #19) narrows detect_intent's classified
  question types to exactly one: when the classifier returned several, it
  asks which one the rep means (their descriptions plus "Something else",
  which resolves to `unclassified`), so every turn follows one predictable
  doc-type retrieval path. Runs after the family question, so a turn
  needing both asks family first, then question type -- once each.

lookup_document: a rep asking for a document ("pull the REFLEX HYBRID
  tray overhead guide") wants the document, not a summary of it. When one
  document in the turn's family clearly matches by filename, document
  type, family and tray tags (agents/tools/document_lookup.py's
  single_best_match), the answer is a fixed sentence citing it -- no
  retrieval, no model call, no fact check -- and the citation opens the
  whole PDF in the viewer.

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

Tray (issue #21) is a soft signal only: resolve_query_parts derives the
turn's trays from the Parts the rep's own words match and, for Cannulated
Screws, from a stated screw diameter; rerank gives documents tagged with
one of them a small bonus, and each resolved Part's tray reaches generate
in the catalog facts. Never asked for, and never a retrieval filter --
most families have one tray, and real questions span trays.

Earlier turns reach generate only as thread memory (issue #7,
memory/thread_memory.py): finalize appends one small record per turn, and
generate renders them into a token-capped "Earlier in this conversation"
block instead of re-sending the whole message history. resolve_skus looks
the previous turn's remembered SKUs up again, so a follow-up naming no SKU
keeps the Parts the last answer was about.
"""
import asyncio
import logging
import math
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, PrivateAttr

from agents.citations import extract_citations, strip_citations
from agents.fact_check import check_answer, format_issues_for_correction
from agents.intent_decisions import decide_intent, decide_or_fall_back, decide_question_type_reply
from agents.question_types import (
    DOCUMENTS_LOOKUP,
    QUESTION_TYPE_NAMES,
    QUESTION_TYPES,
    UNCLASSIFIED,
    QuestionType,
)
from agents.registry import register
from agents.state import BaseAgentState, RetrievedPassage
from agents.tools.document_lookup import single_best_match
from agents.tools.part_lookup import part_lookup
from agents.tools.term_extraction import extract_candidate_terms, with_singular_variants
from agents.tools.vector_search import vector_search
from catalog.catalog_skus import load_catalog_skus
from catalog.display_names import CatalogDisplayNames, catalog_display_names_for
from catalog.family_descriptions import load_family_descriptions
from catalog.repository import ProductFamilyRecord, TrayRecord, list_families_with_trays
from config.llm_clients import get_chat_model
from doc_type_hierarchy.repository import RankedDocumentType, list_hierarchy
from documents.repository import DocumentRecord, list_chunks, list_documents
from memory.thread_memory import (
    build_turn_record,
    remembered_part_lines,
    remembered_parts,
    render_thread_memory,
)
from model_assignments.runtime import SlotModel, invoke_with_output_fallback, slot_model_for_run
from model_assignments.slots import Slot
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
    # Set by resolve_query_parts (issue #21), which runs before rerank:
    # query_parts is the Part lookup for the rep's own query terms --
    # resolve_skus starts from it rather than looking those terms up again
    # -- and turn_tray_ids is the ids of the trays those Parts belong to,
    # plus any tray a stated screw diameter names. rerank's tray bonus
    # reads turn_tray_ids; nothing filters on it.
    query_parts: list[dict]
    turn_tray_ids: list[str]
    # Also set by resolve_query_parts: the display name of every tray in
    # the resolved family (every family's, on an unscoped turn), so
    # self_eval's tray check recognizes a tray the answer names even when
    # no resolved Part is in it.
    family_tray_names: list[str]
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
    # detected_question_types is the classifier's raw answer (deduped, with
    # `unclassified` dropped when anything else is present) -- possibly
    # several types. clarify_question_type narrows it to
    # resolved_question_type, exactly one value (issue #19), asking the rep
    # when there were several; that one value drives hybrid_retrieve's
    # document-type filter (the doc type hierarchy table, issue #17),
    # rerank's bonus, and generate's procedural branch. Always one of
    # agents/question_types.py's QUESTION_TYPE_NAMES -- the classifier's
    # Literal field and _resolve_question_type_reply both enforce it.
    # Unlike search_query and the clarification fields, the API layer doesn't
    # reset resolved_family/resolved_family_id: the next turn's
    # detect_intent reads them back as the thread's previous family (issue
    # #22), so a follow-up naming no family doesn't ask again.
    resolved_family: str | None
    resolved_family_id: str | None
    detected_question_types: list[str]
    resolved_question_type: str
    # Set by hybrid_retrieve from the doc type hierarchy table: document
    # type id -> rerank's additive priority bonus for this turn's question
    # type. Carried to rerank in state so the table is read once per turn.
    document_type_bonuses: dict[str, float]
    # Set by lookup_document, which only runs on a documents_lookup turn:
    # the id of the document it answered with, or None when no document
    # clearly matched and the turn carries on through retrieval. Always
    # written when the node runs, and only read right after it, so a
    # previous turn's value never decides anything.
    looked_up_document_id: str | None


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
    # What lets detect_intent tell a follow-up that names no family (stay on
    # the previous one) from one naming a different family ambiguously (ask)
    # -- both come back with product_family null (issue #22). Defaults to
    # False, but that only matters with a previous family to stay on: a
    # first turn asks exactly as before whatever this says.
    mentions_family: bool = Field(
        default=False,
        description=(
            "True if the question itself names or points at any product "
            "family or brand, even ambiguously (e.g. 'the REFLEX tray'); "
            "false if it names none at all, like a follow-up 'what about "
            "the 3.5 PT?'."
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
            "questions are exactly one, but include more than one when the "
            "question could be read as more than one kind of thing (the rep "
            "is then asked which they mean). Use unclassified, alone, if "
            "none of the others clearly applies."
        ),
    )
    # A decision model's most likely families with their probabilities
    # (issue #30), which the intent eval saves per row; a chat model
    # reports none. Private, so it never joins the structured-output schema.
    _family_probabilities: dict[str, float] = PrivateAttr(default_factory=dict)

    @property
    def family_probabilities(self) -> dict[str, float]:
        return self._family_probabilities


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


def _family_for_turn(
    result: "_IntentClassification",
    families: list[ProductFamilyRecord],
    display_names: CatalogDisplayNames,
    previous_family: ProductFamilyRecord | None,
) -> ProductFamilyRecord | None:
    """detect_intent's family pick before any clarification: the
    classifier's family if it's a real one, else the thread's previous
    family when the new question names no family at all (issue #22). Shared
    with the intent eval, which replays a multi-turn row's earlier turns
    through it.
    """
    family = _resolve_family(result.product_family, families, display_names)
    if family is None and previous_family is not None and not result.mentions_family:
        return previous_family
    return family


def _needs_family_clarification(
    family: ProductFamilyRecord | None, families: list[ProductFamilyRecord]
) -> bool:
    """detect_intent's pause condition: the classifier named no real family
    while there are families to choose between. Its own function so the
    intent eval (evals/intent_eval.py, issue #25) scores the same decision
    the graph makes rather than a copy of it.
    """
    return family is None and bool(families)


def _clarification_options(
    candidate_names: list[str],
    families: list[ProductFamilyRecord],
    display_names: CatalogDisplayNames,
    previous_family: ProductFamilyRecord | None = None,
) -> list[str]:
    """The family display names a clarification offers: the classifier's
    candidates that are real families, deduped and capped at
    MAX_FAMILY_CANDIDATES -- or every family when none of them is. The
    thread's previous family, when there is one, always comes first (issue
    #22): a follow-up naming another family ambiguously may still mean it.
    """
    leading = [previous_family.display_name] if previous_family else []
    candidates: dict[str, None] = {}
    for name in candidate_names:
        family = _resolve_family(name, families, display_names)
        if family is not None:
            candidates.setdefault(family.display_name, None)
    if candidates:
        return list(dict.fromkeys([*leading, *candidates]))[:MAX_FAMILY_CANDIDATES]
    return list(dict.fromkeys([*leading, *(family.display_name for family in families)]))


def _previous_family(
    state: DeterministicState, families: list[ProductFamilyRecord]
) -> ProductFamilyRecord | None:
    """The family the thread's last turn resolved to (issue #22): the
    checkpointed resolved_family_id, which the API layer doesn't reset
    between turns, looked up by id so an admin rename since then shows its
    current display name. None on a thread's first turn, after a turn that
    ran unscoped, or if that family is no longer mirrored.
    """
    previous_family_id = state.get("resolved_family_id")
    return next((family for family in families if str(family.id) == previous_family_id), None)


def _previous_user_message(state: DeterministicState) -> str | None:
    """The rep's message from the turn before this one: the last
    HumanMessage before this turn's own, which the API layer has already
    appended to `messages`. Resuming a clarification adds no HumanMessage,
    so this is always the previous turn's question.
    """
    user_messages = [message for message in state.get("messages") or [] if isinstance(message, HumanMessage)]
    return user_messages[-2].text if len(user_messages) >= 2 else None


async def _classify_intent(
    query: str,
    family_names: list[str],
    previous_family: str | None = None,
    previous_message: str | None = None,
    *,
    family_descriptions: dict[str, str] | None = None,
    slot_model: SlotModel | None = None,
) -> _IntentClassification:
    """`slot_model` defaults to the intent-classification slot of the graph
    run this is called from (issue #24); the intent eval passes its own.
    `family_descriptions` is display name -> the family's generated
    description (catalog/family_descriptions.py, issue #30), which both
    paths show beside its name.

    A decision model in the slot (issue #23) answers through
    agents/intent_decisions.py instead, in the same shape. If that call
    fails, the slot's default chat path below runs.
    """
    family_descriptions = family_descriptions or {}
    slot_model = slot_model or slot_model_for_run(Slot.INTENT_CLASSIFICATION)
    if slot_model.is_decision_model:
        decision = await decide_or_fall_back(
            slot_model,
            lambda client: decide_intent(
                client,
                slot_model,
                query,
                family_names,
                previous_family,
                previous_message,
                MAX_FAMILY_CANDIDATES,
                family_descriptions,
            ),
        )
        if decision is not None:
            classification = _IntentClassification(
                product_family=decision.product_family,
                candidate_families=decision.candidate_families,
                mentions_family=decision.mentions_family,
                question_type=decision.question_types,
            )
            classification._family_probabilities = decision.family_probabilities
            return classification
        slot_model = SlotModel(slot_model.slot)
    family_listing = "\n".join(
        f"- {name}: {family_descriptions[name]}" if family_descriptions.get(name) else f"- {name}"
        for name in family_names
    )
    question_type_listing = "\n".join(f"- {question_type.name}: {question_type.description}" for question_type in QUESTION_TYPES)
    # Hints only (issue #22): product_family still comes from what the new
    # question itself says, and detect_intent -- not the model -- decides
    # that a question naming no family stays on the previous one.
    follow_up_context = ""
    if previous_message:
        follow_up_context += f"\n\nThe rep's previous question in this conversation: {previous_message}"
    if previous_family:
        follow_up_context += f"\nThat question was about the {previous_family} product family."
    if follow_up_context:
        follow_up_context += (
            "\nUse this only to understand what a follow-up refers to. "
            "Classify product_family from what the new question names or "
            "implies, and set mentions_family to false if it names or points "
            "at no product family at all."
        )
    prompt = [
        SystemMessage(
            content=(
                "Classify the sales rep's question below.\n\n"
                "Allowed product families, each with its trays and "
                "distinctive parts from the catalog:\n"
                f"{family_listing or '(none configured)'}\n"
                f"Allowed question types:\n{question_type_listing}\n\n"
                "Set product_family to null if the question doesn't "
                "clearly name or imply exactly one of the allowed "
                "product families, by its name or by a tray or part "
                "listed for it -- when genuinely unclear, always "
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
                "type, output multiple types; if none of the types "
                "clearly fits, output unclassified alone."
                f"{follow_up_context}"
            )
        ),
        {"role": "user", "content": query},
    ]
    return await invoke_with_output_fallback(
        slot_model,
        get_chat_model,
        lambda chat_model: chat_model.with_structured_output(_IntentClassification).ainvoke(prompt),
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

    On a follow-up turn (issue #22), the thread's previous family
    (_previous_family) and the rep's previous message go to the classifier
    as hints. A follow-up the classifier says names no family at all
    (mentions_family false) silently stays on the previous family: no
    interrupt and no "still on X" note, since the answer's sources already
    show which family it's about. One clearly naming a different family
    switches to it; one naming a family ambiguously still asks, offering
    the previous family first. The question type is always classified
    fresh. #7's thread memory (Ticket A) builds on this rather than
    replacing it: each turn's record notes its family for generate's
    "Earlier in this conversation" block, but which family a follow-up
    stays on is still decided here, from resolved_family_id.

    Families come from the Postgres catalog mirror (catalog.repository),
    by display name, so an admin rename shows up here with no code change.
    The classifier sees each with its description, generated from the
    catalog's trays and Parts and cached (catalog/family_descriptions.py,
    issue #30). If none are mirrored yet (a fresh environment before the catalog
    seed), there's nothing to disambiguate against, so this skips straight
    past the interrupt and leaves resolved_family None.

    question_type never triggers this node's interrupt: the classified
    types go out as detected_question_types, and the next node,
    clarify_question_type, owns narrowing them to one (issue #19). Keeping
    that a separate node is what keeps its interrupt from re-running this
    node's classifier call on resume, and puts it after the family question.
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

    previous_family = _previous_family(state, families)
    result = await _classify_intent(
        state["query"],
        [family.display_name for family in families],
        previous_family=previous_family.display_name if previous_family else None,
        previous_message=_previous_user_message(state),
        family_descriptions=await load_family_descriptions(families) if families else {},
    )
    family = _family_for_turn(result, families, display_names, previous_family)

    if _needs_family_clarification(family, families):
        # Everything above re-runs on resume (LangGraph replays the node
        # from its start), the classifier call included.
        answer = interrupt(
            {
                "question": "Which product family is this about?",
                "options": _clarification_options(
                    result.candidate_families, families, display_names, previous_family
                ),
            }
        )
        family = _resolve_family(answer, families, display_names)

    return {
        "resolved_family": family.display_name if family else None,
        "resolved_family_id": str(family.id) if family else None,
        "detected_question_types": _detected_question_types(result.question_type),
    }


# The clarification option that resolves to UNCLASSIFIED (issue #19): the
# rep's way of saying none of the detected readings is what they meant.
SOMETHING_ELSE = "Something else"


def _detected_question_types(classified: list[str]) -> list[str]:
    """The classifier's question types, deduped in order. `unclassified`
    only means anything alone, so it's dropped when a real type came back
    with it; nothing at all comes back as `unclassified`.
    """
    detected = [
        question_type for question_type in dict.fromkeys(classified) if question_type != UNCLASSIFIED
    ]
    return detected or [UNCLASSIFIED]


def _needs_question_type_clarification(detected_question_types: list[str]) -> bool:
    """clarify_question_type's pause condition: more than one detected type.
    Shared with the intent eval, like _needs_family_clarification.
    """
    return len(detected_question_types) > 1


def _question_type_option(question_type: QuestionType) -> str:
    # str.capitalize() would lowercase the rest of the description.
    return question_type.description[:1].upper() + question_type.description[1:]


def _match_question_type_reply(reply: str) -> str | None:
    """A clicked option, or free text that names a question type outright
    (its option text, or its slug with or without underscores), case-
    insensitively. None for anything else.
    """
    normalized = " ".join(reply.split()).lower()
    if normalized == SOMETHING_ELSE.lower():
        return UNCLASSIFIED
    for question_type in QUESTION_TYPES:
        spellings = {
            question_type.name,
            question_type.name.replace("_", " "),
            _question_type_option(question_type).lower(),
        }
        if normalized in spellings:
            return question_type.name
    return None


class _QuestionTypeReply(BaseModel):
    question_type: Literal[*QUESTION_TYPE_NAMES] = Field(
        description=(
            "The one allowed question type the rep's reply says their "
            "question is, or unclassified if the reply doesn't make it any "
            "of them."
        ),
    )


async def _classify_question_type_reply(query: str, reply: str) -> str:
    """Free text that doesn't name a question type outright ("I need the
    steps") still has to come back as exactly one, so the model picks it --
    from every question type, not only the offered ones, like the family
    clarification's free text. Only runs on that free-text path. A decision
    model in the slot answers it as a Choice (issue #23).
    """
    slot_model = slot_model_for_run(Slot.INTENT_CLASSIFICATION)
    if slot_model.is_decision_model:
        decided = await decide_or_fall_back(
            slot_model,
            lambda client: decide_question_type_reply(client, slot_model, query, reply),
        )
        if decided is not None:
            return decided
        slot_model = SlotModel(slot_model.slot)
    question_type_listing = "\n".join(f"- {question_type.name}: {question_type.description}" for question_type in QUESTION_TYPES)
    prompt = [
        SystemMessage(
            content=(
                "A sales rep was asked what kind of question they're asking. "
                "Pick the one question type their reply means.\n\n"
                f"Allowed question types:\n{question_type_listing}"
            )
        ),
        {"role": "user", "content": f"Question: {query}\nWhat they meant: {reply}"},
    ]
    result = await invoke_with_output_fallback(
        slot_model,
        get_chat_model,
        lambda chat_model: chat_model.with_structured_output(_QuestionTypeReply).ainvoke(prompt),
    )
    return result.question_type


async def clarify_question_type(state: DeterministicState) -> dict:
    """Narrows detect_intent's detected_question_types to exactly one
    resolved_question_type (issue #19), so retrieval follows one doc-type
    path instead of a union of several. One detected type (including a
    plain `unclassified`) resolves straight through; several pause the
    graph via interrupt() offering their descriptions plus SOMETHING_ELSE,
    resumed through the same POST /chat/resume as the family question.

    A clicked option or a reply naming a type resolves with no model call
    (_match_question_type_reply); other free text goes to
    _classify_question_type_reply. Either way the result is a single type,
    so there's never a second question-type clarification in a turn.

    A separate node from detect_intent so resuming here replays only this
    node -- not the intent classifier -- and so it runs after the family
    question. It has its own budget: the synonym clarification's
    clarification_rounds doesn't count it, and it doesn't count that.
    """
    detected = state.get("detected_question_types") or [UNCLASSIFIED]
    if not _needs_question_type_clarification(detected):
        return {"resolved_question_type": detected[0]}

    question_types_by_name = {question_type.name: question_type for question_type in QUESTION_TYPES}
    reply = interrupt(
        {
            "question": "Which of these best describes what you're asking?",
            "options": [
                *(_question_type_option(question_types_by_name[name]) for name in detected),
                SOMETHING_ELSE,
            ],
        }
    )
    reply = (reply or "").strip()
    if not reply:
        return {"resolved_question_type": UNCLASSIFIED}
    resolved = _match_question_type_reply(reply)
    if resolved is None:
        resolved = await _classify_question_type_reply(state["query"], reply)
    return {"resolved_question_type": resolved}


def _after_question_type(
    state: DeterministicState,
) -> Literal["lookup_document", "resolve_synonyms"]:
    if state.get("resolved_question_type") == DOCUMENTS_LOOKUP:
        return "lookup_document"
    return "resolve_synonyms"


def _document_lookup_answer(document: DocumentRecord, citation: str) -> str:
    """The fixed sentence a looked-up document comes back as: its filename,
    and its document type and family when tagged, with the citation that
    opens it in the viewer."""
    tags = ", ".join(
        filter(None, [document.document_type_name, document.product_family_display_name])
    )
    described = f"**{document.filename}** ({tags})" if tags else f"**{document.filename}**"
    return f"Here's {described}. Open it below to view the whole document. [{citation}]"


async def lookup_document(state: DeterministicState) -> dict:
    """The documents_lookup fast path: a rep asking for a document wants
    the document itself, which retrieval and generate used to answer with
    a summary of whatever chunks ranked highest. Matches the request
    against every indexed document's filename and document type, family
    and tray tags (single_best_match), within the turn's family when one
    resolved. One clear winner answers the turn with no model call; a tie,
    a thin match or no match leaves looked_up_document_id None and the
    turn goes through retrieval and generate as before.

    The answer cites the document's first chunk, which is where the
    citation viewer opens it. It skips self_eval: the sentence is built
    from the document's own record, so there's nothing to fact-check.
    """
    family_id = state.get("resolved_family_id")
    documents = [
        document
        for document in await list_documents()
        if document.status == "done"
        and (family_id is None or str(document.product_family_id) == family_id)
    ]
    document = single_best_match(state["query"], documents)
    chunks = await list_chunks(document.id) if document else []
    if document is None or not chunks:
        return {"looked_up_document_id": None}
    citation = f"{document.id}#{chunks[0].chunk_index}"
    return {
        "looked_up_document_id": str(document.id),
        "answer": _document_lookup_answer(document, citation),
        "citations": [citation],
    }


def _after_document_lookup(state: DeterministicState) -> Literal["finalize", "resolve_synonyms"]:
    return "finalize" if state.get("looked_up_document_id") else "resolve_synonyms"


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
# rerank's bonus for a chunk whose document is tagged with one of the
# turn's trays (issue #21) -- the same size as the *lowest* doc-type rank's
# bonus (1 * _DOCTYPE_BONUS_STEP), never more, so the doc-type order still
# dominates: tray only breaks ties among otherwise-close passages. It's a
# soft signal because most families have a single tray and real questions
# span trays (a Lapidus back table needs the Nitinol and the MINI tray).
# Flat, not per matching tray: a document carrying all of a family's trays
# shouldn't outrank one carrying just the right one.
_TRAY_BONUS = _DOCTYPE_BONUS_STEP


def _ranked_document_type_ids(
    hierarchy: dict[str, list[RankedDocumentType]],
) -> dict[str, list[str]]:
    return {
        question_type: [str(document_type.id) for document_type in document_types]
        for question_type, document_types in hierarchy.items()
    }


def _document_type_bonuses(
    question_type: str | None, ranking: dict[str, list[str]]
) -> dict[str, float]:
    """Document type id -> rerank bonus for this turn's question type, in
    its ranked order (most authoritative gets the biggest bonus); a doc type
    it doesn't rank gets none. Empty when it has nothing mapped.
    """
    ranked_ids = ranking.get(question_type or "") or []
    return {
        document_type_id: (len(ranked_ids) - rank) * _DOCTYPE_BONUS_STEP
        for rank, document_type_id in enumerate(ranked_ids)
    }


def _allowed_document_type_ids(
    question_type: str | None, ranking: dict[str, list[str]]
) -> list[str] | None:
    """The document type ids hybrid_retrieve hard-filters to: the ones
    mapped to this turn's question type. Returns None (no filter --
    retrieve unfiltered) when it has nothing mapped (e.g. "unclassified" or
    "documents_lookup"), since restricting to an empty list would be worse
    than not filtering at all when there's no actual rule to apply.
    """
    return list(ranking.get(question_type or "") or []) or None


async def hybrid_retrieve(state: DeterministicState) -> dict:
    # Hard-filtered by detect_intent's resolved family (issue #18): a
    # family-scoped turn only sees chunks of documents tagged with that
    # family, and a document with no family is excluded -- accepted, since
    # the indexing gate (issue #14) requires a family before a document is
    # indexed. No family resolved -> no family filter. That also drops a
    # document describing the identical Part under another family (174
    # SKUs sit in several families' trays); relaxing the filter to every
    # family holding the turn's resolved SKUs was left out of issue #21 as
    # a candidate follow-up ticket. Never filtered by tray -- see rerank.
    search_query = state.get("search_query") or state["query"]
    # Canonical terms, not a flat canonical+alias mix -- see
    # resolve_synonyms' docstring.
    search_text = " ".join(filter(None, [search_query, *state.get("resolved_canonical_terms", [])]))
    question_type = state.get("resolved_question_type")
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
    allowed_document_type_ids = _allowed_document_type_ids(question_type, ranking)
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
            "tray_ids": result.get("tray_ids") or [],
        }
        for result in results
    ]
    return {
        "retrieved": retrieved,
        "document_type_bonuses": _document_type_bonuses(question_type, ranking),
    }


# Issue #21's stated-diameter rule applies to this one family (its catalog
# name): Cannulated Screws' four trays split cleanly by screw diameter
# (Forefoot Fixation 2.0-3.5, 4.0/4.5, 5.5, 6.5/7.0), so a diameter the rep
# states names the tray outright. Other families mix screw sizes within a
# tray, where a stated size says nothing about which one.
_DIAMETER_RULE_FAMILY = "Cannulated Screws"
# A size as a rep writes it: "4.0", "4.5mm", "Ø6.5", "7 mm". The number
# can't touch a letter or digit on either side, so a SKU (MSD14020) never
# yields one, and "4.0/4.5" yields both. A length ("40mm") matches too, but
# no screw has a 40mm diameter, so it names no tray.
_STATED_SIZE_PATTERN = re.compile(r"(?<![A-Za-z0-9.])(\d{1,2}(?:\.\d+)?)(?:\s*mm)?(?![A-Za-z0-9])")
# A screw's diameter is the number before the first " X " in its catalog
# description: "SCREW, CANN, HEADED (T15), 4.0 X 20MM" -> 4.0, and
# "SCREW, CANN, HEADLESS, 6.5 X 40 X 16MM" -> 6.5.
_SCREW_DIAMETER_PATTERN = re.compile(r"^SCREW\b.*?\b(\d+(?:\.\d+)?) X ", re.IGNORECASE)


def _screw_diameters(descriptions: list[str]) -> set[float]:
    matches = (_SCREW_DIAMETER_PATTERN.match(description) for description in descriptions)
    return {float(match.group(1)) for match in matches if match}


def _trays_for_stated_diameters(query: str, descriptions_by_tray: dict[str, list[str]]) -> list[str]:
    """The trays holding a screw in a diameter the query states, read off
    each tray's own Part descriptions (graph_client.
    list_part_descriptions_by_tray) -- the rule is data-driven, so there's
    no diameter -> tray table in code to fall out of step with the catalog.
    Empty when the query states no size, or none any tray's screws come in.
    """
    stated = {float(size) for size in _STATED_SIZE_PATTERN.findall(query)}
    if not stated:
        return []
    return sorted(
        tray for tray, descriptions in descriptions_by_tray.items() if stated & _screw_diameters(descriptions)
    )


def _turn_tray_ids(parts: list[dict], diameter_tray_names: list[str], trays: list[TrayRecord]) -> list[str]:
    """The ids of the trays in `trays` (the resolved family's, or every
    family's on an unscoped turn) that this turn points at: each matched
    Part's trays (display names, as part_lookup returns them) plus the
    stated-diameter rule's (catalog names, straight from the graph). Order
    follows the Parts', deduped; a name with no mirrored tray is dropped.
    """
    ids_by_display_name = {tray.display_name: str(tray.id) for tray in trays}
    ids_by_catalog_name = {tray.name: str(tray.id) for tray in trays}
    tray_ids = [
        ids_by_display_name.get(tray_name)
        for part in parts
        for tray_name in part.get("trays") or []
    ] + [ids_by_catalog_name.get(tray_name) for tray_name in diameter_tray_names]
    return list(dict.fromkeys(tray_id for tray_id in tray_ids if tray_id))


def _query_terms(state: DeterministicState) -> list[str]:
    """The rep's own query terms plus this turn's canonical terms -- e.g. a
    query that says "wire" already resolved "guidepin", the term that
    actually substring-matches Part descriptions."""
    return with_singular_variants(
        [*extract_candidate_terms(state["query"]), *state.get("resolved_canonical_terms", [])]
    )[:MAX_TERMS_PER_TURN]


async def _lookup_parts(terms: list[str], family: str | None) -> list[dict]:
    """Every Part `terms` match, deduped by SKU in term order."""
    results = await asyncio.gather(
        *(part_lookup.ainvoke({"term": term, "product_family": family}) for term in terms)
    )
    matched: dict[str, dict] = {}
    for parts in results:
        for part in parts:
            sku = part.get("sku")
            if sku:
                matched.setdefault(sku, part)
    return list(matched.values())


async def resolve_query_parts(state: DeterministicState) -> dict:
    """Issue #21: looks up the Parts the rep's own query terms match, and
    derives the turn's trays from them, before rerank -- which runs ahead
    of resolve_skus, so the trays have to be known earlier than that node
    could tell it. resolve_skus then starts from these query_parts and
    only adds the passage-derived lookup technique_procedural turns get.

    A turn's trays are each matched Part's trays (the graph's
    BELONGS_TO_TRAY, via part_lookup) plus, for Cannulated Screws only, any
    tray a stated screw diameter names (_trays_for_stated_diameters -- a
    rule, not a model call). They only feed rerank's _TRAY_BONUS: tray is
    never asked for and never filters retrieval, since most families have
    one tray and a real question can span several.

    Part lookup stays scoped to the resolved family, so a Part shared with
    another family (MPSN3522 sits in 7 families' trays) only contributes
    that family's trays here -- relaxing #18's family scoping for shared
    Parts is a candidate follow-up ticket, not part of this one.
    """
    # A display name -- part_lookup maps it back to the catalog name.
    family_name = state.get("resolved_family")
    query_parts = await _lookup_parts(_query_terms(state), family_name)

    families = await list_families_with_trays()
    family = next(
        (candidate for candidate in families if str(candidate.id) == state.get("resolved_family_id")), None
    )
    trays = family.trays if family else [tray for every_family in families for tray in every_family.trays]
    diameter_tray_names: list[str] = []
    if family is not None and family.name == _DIAMETER_RULE_FAMILY and _STATED_SIZE_PATTERN.search(state["query"]):
        descriptions_by_tray = await get_graph_client().list_part_descriptions_by_tray(family.name)
        diameter_tray_names = _trays_for_stated_diameters(state["query"], descriptions_by_tray)

    return {
        "query_parts": query_parts,
        "turn_tray_ids": _turn_tray_ids(query_parts, diameter_tray_names, trays),
        "family_tray_names": [tray.display_name for tray in trays],
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


def _tray_bonus(passage_tray_ids: list[str], turn_tray_ids: set[str]) -> float:
    return _TRAY_BONUS if turn_tray_ids.intersection(passage_tray_ids) else 0.0


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

    Issue #21 adds _TRAY_BONUS on top for a passage whose document is
    tagged with one of resolve_query_parts' turn_tray_ids -- a tiebreak,
    not a filter.
    """
    passages = state["retrieved"]
    if not passages:
        return {"reranked": []}
    query_terms = [state["query"], *state.get("resolved_canonical_terms", [])]
    documents = [passage["text"] for passage in passages]
    bm25 = _normalize_to_unit_range(_bm25_scores(query_terms, documents))
    # Doc type id -> bonus, computed by hybrid_retrieve from the hierarchy
    # table for this turn's question type. An untagged chunk (no id) or a
    # type that question type doesn't rank gets no bonus.
    document_type_bonuses = state.get("document_type_bonuses") or {}
    turn_tray_ids = set(state.get("turn_tray_ids") or [])
    # score is overwritten with the combined (BM25 + doctype priority
    # bonus + tray bonus) value, not left at hybrid_retrieve's original RRF fusion
    # score -- that's the number that actually decided this passage's
    # rank, and a trace/debugging view showing the old, pre-rerank score
    # here would misrepresent why a given passage made the cut.
    combined = [
        {
            **passage,
            "score": score
            + document_type_bonuses.get(passage.get("document_type_id") or "", 0.0)
            + _tray_bonus(passage.get("tray_ids") or [], turn_tray_ids),
        }
        for passage, score in zip(passages, bm25)
    ]
    # Ties broken by the doc-type bonus alone: _TRAY_BONUS can lift a
    # passage level with the next doc-type rank up, never past it.
    scored = sorted(
        combined,
        key=lambda passage: (
            passage["score"],
            document_type_bonuses.get(passage.get("document_type_id") or "", 0.0),
        ),
        reverse=True,
    )
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

    Issue #21 moved the query-term half of that lookup upstream again, to
    resolve_query_parts -- rerank needs those Parts' trays and runs before
    this node. It's still only a Part lookup, not search-text enrichment,
    so the 2026-09-04 problem doesn't come back. This node starts from its
    query_parts and adds only the passage-derived lookup, so no query term
    is looked up twice, and the two together stay within
    MAX_TERMS_PER_TURN, query terms first.

    Issue #7 (thread memory) adds the previous turn's remembered SKUs, on
    every question type, so a follow-up naming no SKU ("and the 3.5
    version?") still has the Parts the last answer was about, with their
    full catalog properties for generate and the fact check. They're
    looked up after the query terms and before passage terms, inside the
    same MAX_TERMS_PER_TURN, and scoped to this turn's family like the
    rest: a follow-up that switched family doesn't drag the old one's
    Parts along.
    """
    query_parts = list(state.get("query_parts") or [])
    query_terms = _query_terms(state)
    # A display name -- part_lookup maps it back to the catalog name.
    family = state.get("resolved_family")
    previous_skus = _previous_turn_skus(state, {part["sku"] for part in query_parts})[
        : max(MAX_TERMS_PER_TURN - len(query_terms), 0)
    ]
    previous_parts = await _lookup_exact_skus(previous_skus, family)

    matched = {part["sku"]: part for part in query_parts}
    for part in previous_parts:
        matched.setdefault(part["sku"], part)
    if state.get("resolved_question_type") != "technique_procedural":
        return {"resolved_parts": list(matched.values())}

    passage_text = " ".join(passage["text"] for passage in (state.get("reranked") or []))
    passage_terms = [
        term
        for term in with_singular_variants(extract_candidate_terms(passage_text))
        if term not in query_terms
    ][: max(MAX_TERMS_PER_TURN - len(query_terms) - len(previous_skus), 0)]
    passage_parts = await _lookup_parts(passage_terms, family)

    for part in passage_parts:
        matched.setdefault(part["sku"], part)
    return {"resolved_parts": list(matched.values())}


def _previous_turn_skus(state: DeterministicState, already_resolved: set[str]) -> list[str]:
    """The SKUs the thread's last turn remembered (memory/thread_memory.py),
    minus any this turn already resolved. Empty on a thread's first turn.
    """
    records = state.get("turn_memory") or []
    if not records:
        return []
    return [part["sku"] for part in records[-1].get("parts") or [] if part["sku"] not in already_resolved]


async def _lookup_exact_skus(skus: list[str], family: str | None) -> list[dict]:
    """The Part each SKU names exactly. part_lookup also matches a term
    against descriptions, which a remembered SKU has no reason to pull in.
    """
    results = await asyncio.gather(
        *(part_lookup.ainvoke({"term": sku, "product_family": family}) for sku in skus)
    )
    exact_matches = (
        next((part for part in parts if (part.get("sku") or "").upper() == sku.upper()), None)
        for sku, parts in zip(skus, results)
    )
    return [part for part in exact_matches if part is not None]


def _format_part(part: dict) -> str:
    sku = part.get("sku", "?")
    description = part.get("description", "")
    properties = {
        key: value
        for key, value in part.items()
        if key not in ("sku", "description", "trays") and value not in (None, "")
    }
    # Issue #21: the Part's tray, so an answer can say "in the REFLEX MINI
    # tray" -- and the fact check can hold it to that. A part in several of
    # the family's trays lists each; "; " because a tray name can itself
    # contain " | " ("Soft Tissue | Comprehensive Instrument Tray").
    if part.get("trays"):
        properties["tray"] = "; ".join(part["trays"])
    extra = ", ".join(f"{key}={value}" for key, value in properties.items())
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
    "- Before answering, compare every specific value (SKU, dimension, "
    "length range, material, thread type, color, driver, quantity, tray) "
    "that both the known catalog facts and a context passage state for the "
    "*same* item. When they disagree, always answer with the catalog value "
    "-- it's pulled directly from the parts database, while a passage came "
    "through OCR/text extraction or an older document revision and can "
    "carry its own errors.\n"
    "- Never resolve such a disagreement silently. End the answer with a "
    "short section headed \"Conflicting sources\", one line per "
    "disagreement, each line naming the item, the catalog value, and the "
    "passage's value with its bracketed chunk id, e.g. \"MSCF3520: the "
    "catalog lists Full Thread; [doc-id#3] says partial thread. This answer "
    "uses the catalog value.\" Keep each disagreement on its own single "
    "line. Only list genuine contradictions about the same item -- not a "
    "passage that's merely less complete, a different item, or a value the "
    "catalog doesn't record. Leave the section out when there are none.\n"
    "- When the question asks where an item is, or which tray or set it's "
    "in, name its tray from the known catalog facts (their tray= value), "
    "exactly as written there.\n"
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
# turn as "technique_procedural" -- a spec-lookup answer
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
    # Earlier turns reach the model only as thread memory (issue #7): one
    # "Earlier in this conversation" block of per-turn records, capped at
    # THREAD_MEMORY_TOKEN_CAP -- not state["messages"], which re-sent every
    # earlier Q&A pair verbatim and grew the prompt by one per turn.
    # `messages` stays the transcript the UI shows. Order is system prompt,
    # then memory, then this turn's facts, passages and question, so the
    # system prompt stays a stable prefix for prompt caching. This is a
    # *draft*: on a clarification round, generate runs again with an
    # enriched search (see request_clarification), and on a correction
    # pass it runs again with self_eval's fact-check issues -- only the
    # final version becomes permanent history and memory, in finalize.
    memory_block, memory_block_tokens = render_thread_memory(state.get("turn_memory") or [])
    clarification_reply = state.get("clarification_reply")
    clarification_note = (
        f"\n\nThe rep clarified: {clarification_reply}" if clarification_reply else ""
    )
    augmented_question = HumanMessage(
        content=(
            (f"{memory_block}\n\n" if memory_block else "")
            + f"Known catalog facts:\n{_format_known_facts(state.get('aggregated_facts', ''))}"
            f"\n\nContext:\n{_format_context(passages)}\n\nQuestion: {state['query']}"
            f"{clarification_note}"
        )
    )
    system_prompt = _GENERATE_SYSTEM_PROMPT
    if state.get("resolved_question_type") == "technique_procedural":
        system_prompt += _PROCEDURAL_FORMAT_ADDENDUM
    prompt = [SystemMessage(content=system_prompt), augmented_question]
    # A correction pass (routed here by _should_correct): show the model
    # its own failed draft and exactly what the fact check found, rather
    # than re-asking from scratch and hoping it lands differently.
    issues = state.get("fact_check_issues") or []
    update: dict = {"memory_tokens": {"thread_memory": memory_block_tokens}}
    if issues:
        prompt += [AIMessage(content=state["answer"]), HumanMessage(content=format_issues_for_correction(issues))]
        update["correction_rounds"] = state.get("correction_rounds", 0) + 1
    # An assigned model's empty answer gets one retry on CHAT_MODEL (issue
    # #24). The call streams through astream_events whichever model answers.
    response = await invoke_with_output_fallback(
        slot_model_for_run(Slot.DETERMINISTIC_ANSWER),
        get_chat_model,
        lambda chat_model: chat_model.ainvoke(prompt),
        is_usable=lambda candidate: bool(candidate.text.strip()),
    )
    answer = response.text
    return {**update, "answer": answer, "citations": extract_citations(answer)}


async def self_eval(state: DeterministicState) -> dict:
    """Fact-checks the draft (agents/fact_check.py) against everything
    generate was actually allowed to use -- the reranked passages *and* the
    resolved catalog Part records. Both matter for the same reason they
    did when this was an LLM judge (ticket 20/21/22's eval run): a claim
    correctly grounded in the catalog but absent from every passage is the
    graph-grounding ticket 20 exists for, not an unsupported claim.

    The whole catalog's SKUs (cached, issue #29) only tell the check what
    counts as a SKU, so an alpha-only one like SNAPWIRE gets checked too.

    Thread memory's remembered Parts (issue #7) count as sources too: the
    previous turn's are already in resolved_parts (resolve_skus looks them
    up again), and every remembered SKU's line is in the source text, so a
    SKU the memory block showed isn't flagged as invented. Earlier turns'
    cited chunk ids are deliberately not citable: a citation still has to
    point at a passage retrieved this turn.
    """
    passages = state.get("reranked") or []
    issues = check_answer(
        state["answer"],
        citation_ids={passage["chunk_id"] for passage in passages},
        source_text="\n".join(
            [
                *(passage["text"] for passage in passages),
                state.get("aggregated_facts") or "",
                remembered_part_lines(state.get("turn_memory") or []),
            ]
        ),
        parts=state.get("resolved_parts") or [],
        tray_names=set(state.get("family_tray_names") or []),
        catalog_skus=await load_catalog_skus(),
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

    Also appends this turn's thread-memory record (issue #7), built from
    state alone with no model call: the question, the answer without its
    citation brackets, the Parts it was about (remembered_parts), its
    cited chunk ids and its family. Here for the same reason as the
    message: only the shipped answer should be remembered.
    """
    answer = state["answer"]
    record = build_turn_record(
        question=state["query"],
        answer=strip_citations(answer),
        parts=remembered_parts(
            answer, state.get("resolved_parts") or [], state.get("query_parts") or []
        ),
        cited_chunk_ids=state.get("citations") or [],
        family=state.get("resolved_family"),
    )
    return {
        "messages": [AIMessage(content=answer, additional_kwargs={"citations": state["citations"]})],
        "turn_memory": [record],
    }


def _should_correct(state: DeterministicState) -> Literal["generate", "finalize"]:
    if state.get("fact_check_issues") and state.get("correction_rounds", 0) < MAX_CORRECTION_ROUNDS:
        return "generate"
    return "finalize"


def build_graph(checkpointer):
    graph = StateGraph(DeterministicState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("clarify_question_type", clarify_question_type)
    graph.add_node("lookup_document", lookup_document)
    graph.add_node("resolve_synonyms", resolve_synonyms)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("resolve_query_parts", resolve_query_parts)
    graph.add_node("rerank", rerank)
    graph.add_node("resolve_skus", resolve_skus)
    graph.add_node("aggregate_facts", aggregate_facts)
    graph.add_node("generate", generate)
    graph.add_node("self_eval", self_eval)
    graph.add_node("request_clarification", request_clarification)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "clarify_question_type")
    graph.add_conditional_edges("clarify_question_type", _after_question_type)
    graph.add_conditional_edges("lookup_document", _after_document_lookup)
    graph.add_conditional_edges("resolve_synonyms", _should_clarify_synonyms)
    graph.add_edge("hybrid_retrieve", "resolve_query_parts")
    graph.add_edge("resolve_query_parts", "rerank")
    graph.add_edge("rerank", "resolve_skus")
    graph.add_edge("resolve_skus", "aggregate_facts")
    graph.add_edge("aggregate_facts", "generate")
    graph.add_edge("generate", "self_eval")
    graph.add_conditional_edges("self_eval", _should_correct)
    graph.add_edge("request_clarification", "resolve_synonyms")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


register("deterministic", build_graph)

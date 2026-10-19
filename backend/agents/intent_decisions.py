"""Intent classification on a decision model (issue #23): the deterministic
workflow's classification questions asked as Choices instead of structured
output from a chat model. Used when the intent-classification slot holds a
decision model (model_assignments/slots.py); an unset slot, or a chat model
there, never reaches this module.

Each decision is one Choice, never a multi-label Noul: Choice
probabilities are well calibrated, and a single threshold per decision is
all 35 labelled rows can tune.

- Product family: one option per family display name, plus UNCLEAR_FAMILY
  (the question points at a family but doesn't settle on one) and
  NO_FAMILY (it names none at all -- what lets a follow-up stay on the
  previous family, issue #22). Each family's criteria carries its generated
  description (catalog/family_descriptions.py, issue #30) -- trays,
  distinctive parts, and which family a shared brand means -- since bare
  names were too thin to choose between. A family is picked only when it's
  the answer with at least the slot's family confidence threshold;
  otherwise the families above CANDIDATE_PROBABILITY_FLOOR, most likely
  first, become the clarification's candidates.
- Question type: one option per QUESTION_TYPES slug, criteria from their
  descriptions. When the runner-up is within the slot's margin of the
  answer, both count as detected, and clarify_question_type asks which
  (issue #19). `unclassified` is never paired: it only means anything
  alone.
- The question-type reply: one Choice over the same options, state
  {question, reply}.

The answers come back shaped like the chat path's, so detect_intent's
pause conditions (_needs_family_clarification and friends) decide exactly
as they do for a chat model, and the intent eval scores this path through
the same code.
"""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from agents.question_types import QUESTION_TYPE_NAMES, QUESTION_TYPES, UNCLASSIFIED
from config.llm_clients import (
    ChoiceAnswer,
    DecisionClient,
    DecisionModelError,
    DecisionResponse,
    get_decision_client,
    record_decision_usage,
)
from model_assignments.runtime import SlotModel, log_slot_output_fallback

ResultType = TypeVar("ResultType")

UNCLEAR_FAMILY = "unclear"
NO_FAMILY = "no_family"
# A family needs at least this probability to be offered as a
# clarification candidate. Not a tuned threshold: it only trims options a
# rep would never mean.
CANDIDATE_PROBABILITY_FLOOR = 0.05
# How many of the family Choice's most likely options IntentDecision
# carries. Only the intent eval reads them: it saves them per row (issue
# #30), so a tuning round doesn't need a dump of its own.
REPORTED_FAMILY_PROBABILITIES = 5

FAMILY_QUESTION_ID = "product_family"
QUESTION_TYPE_QUESTION_ID = "question_type"

_FAMILY_INSTRUCTIONS = (
    "Which product family is `question` about? Pick a family only if `question` clearly "
    "names or implies exactly that one, by its name or by a tray or part its description "
    "lists. Several families can share a brand name (MIS, REFLEX): pick the one that actually matches, and don't switch families on incidental "
    "word overlap -- a question about a screw, wire or guidepin is about the implant or "
    "consumable family, not a powered-instrument family, unless it asks about the powered "
    f"tool itself. Pick {UNCLEAR_FAMILY} if `question` points at a product family or brand "
    f"but doesn't settle on exactly one. Pick {NO_FAMILY} if it names or points at no "
    "product family at all."
)
_FOLLOW_UP_INSTRUCTIONS = (
    " `previous_question` is the rep's previous question in this conversation and "
    "`previous_family` the family it was about: use them only to understand what a "
    f"follow-up refers to. A follow-up that names no family is {NO_FAMILY}."
)
_QUESTION_TYPE_INSTRUCTIONS = (
    "What kind of question is `question`? Most questions are exactly one kind. Pick "
    f"{UNCLASSIFIED} if none of the others clearly applies."
)
_QUESTION_TYPE_REPLY_INSTRUCTIONS = (
    "A sales rep was asked what kind of question `question` is, and answered `reply`. "
    f"Which question type does `reply` mean? Pick {UNCLASSIFIED} if it doesn't make it any "
    "of the others."
)


@dataclass(frozen=True)
class IntentDecision:
    """detect_intent's classification, the same four facts the chat path's
    _IntentClassification carries.
    """

    product_family: str | None
    candidate_families: list[str]
    mentions_family: bool
    question_types: list[str]
    # Display name (or UNCLEAR_FAMILY/NO_FAMILY) -> probability, the
    # REPORTED_FAMILY_PROBABILITIES most likely, for the intent eval.
    family_probabilities: dict[str, float] = field(default_factory=dict)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def family_option_keys(family_names: list[str]) -> dict[str, str]:
    """Option key -> family display name. A Choice key has to be a string
    that isn't a number, and mustn't collide with the two sentinels, so a
    display name that is either gets a prefix.
    """
    keys: dict[str, str] = {}
    for name in family_names:
        key = name
        if name in (UNCLEAR_FAMILY, NO_FAMILY) or _is_number(name):
            key = f"family: {name}"
        keys.setdefault(key, name)
    return keys


def family_question(
    family_keys: dict[str, str],
    *,
    follow_up: bool,
    family_descriptions: dict[str, str] | None = None,
) -> dict:
    """`family_descriptions` is display name -> description
    (catalog/family_descriptions.py); a family without one is described by
    its name alone.
    """
    family_descriptions = family_descriptions or {}
    criteria = {
        key: " ".join(filter(None, [f"The {name} product family.", family_descriptions.get(name)]))
        for key, name in family_keys.items()
    }
    criteria[UNCLEAR_FAMILY] = (
        "The question points at a product family or brand, but not clearly at exactly one."
    )
    criteria[NO_FAMILY] = "The question names or points at no product family at all."
    return {
        "type": "choice",
        "instructions": _FAMILY_INSTRUCTIONS + (_FOLLOW_UP_INSTRUCTIONS if follow_up else ""),
        "criteria": criteria,
    }


def question_type_question(instructions: str = _QUESTION_TYPE_INSTRUCTIONS) -> dict:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {question_type.name: question_type.description for question_type in QUESTION_TYPES},
    }


def pick_family(
    answer: ChoiceAnswer,
    family_keys: dict[str, str],
    threshold: float,
    max_candidates: int,
) -> tuple[str | None, list[str], bool]:
    """(product_family, candidate_families, mentions_family) from the
    family Choice.
    """
    mentions_family = answer.choice != NO_FAMILY
    if answer.choice in family_keys and answer.probabilities.get(answer.choice, 0.0) >= threshold:
        return family_keys[answer.choice], [], mentions_family
    ranked = sorted(
        (
            (probability, key)
            for key, probability in answer.probabilities.items()
            if key in family_keys and probability >= CANDIDATE_PROBABILITY_FLOOR
        ),
        reverse=True,
    )
    candidates = [family_keys[key] for _, key in ranked[:max_candidates]]
    return None, candidates, mentions_family


def top_family_probabilities(answer: ChoiceAnswer, family_keys: dict[str, str]) -> dict[str, float]:
    """The family Choice's REPORTED_FAMILY_PROBABILITIES most likely
    options, keyed by family display name (sentinels as themselves).
    """
    ranked = sorted(answer.probabilities.items(), key=lambda entry: entry[1], reverse=True)
    return {
        family_keys.get(key, key): probability
        for key, probability in ranked[:REPORTED_FAMILY_PROBABILITIES]
    }


def pick_question_types(answer: ChoiceAnswer, margin: float) -> list[str]:
    """The answer, plus the runner-up when it's within `margin` of it --
    two readings close enough that the rep should say which.
    """
    top_probability = answer.probabilities.get(answer.choice, 0.0)
    runners_up = sorted(
        (
            (probability, option)
            for option, probability in answer.probabilities.items()
            if option != answer.choice
        ),
        reverse=True,
    )
    if runners_up and UNCLASSIFIED not in (answer.choice, runners_up[0][1]):
        runner_up_probability, runner_up = runners_up[0]
        if top_probability - runner_up_probability < margin:
            return [answer.choice, runner_up]
    return [answer.choice]


async def _decide(
    client: DecisionClient, slot_model: SlotModel, state: dict, questions: dict
) -> DecisionResponse:
    response = await client.decide(slot_model.model_id, state, questions)
    record_decision_usage(response)
    return response


async def decide_intent(
    client: DecisionClient,
    slot_model: SlotModel,
    query: str,
    family_names: list[str],
    previous_family: str | None,
    previous_message: str | None,
    max_candidates: int,
    family_descriptions: dict[str, str] | None = None,
) -> IntentDecision:
    """Both of detect_intent's decisions in one request. With no families
    configured only the question type is asked: there's nothing to choose
    between, and an empty Choice isn't a question.
    """
    state: dict = {"question": query}
    if previous_message:
        state["previous_question"] = previous_message
    if previous_family:
        state["previous_family"] = previous_family
    family_keys = family_option_keys(family_names)
    questions = {QUESTION_TYPE_QUESTION_ID: question_type_question()}
    if family_keys:
        questions[FAMILY_QUESTION_ID] = family_question(
            family_keys,
            follow_up=bool(previous_message or previous_family),
            family_descriptions=family_descriptions,
        )

    response = await _decide(client, slot_model, state, questions)

    product_family, candidate_families, mentions_family = None, [], False
    family_probabilities: dict[str, float] = {}
    if family_keys:
        family_answer = response.answers[FAMILY_QUESTION_ID]
        product_family, candidate_families, mentions_family = pick_family(
            family_answer, family_keys, slot_model.family_confidence_threshold, max_candidates
        )
        family_probabilities = top_family_probabilities(family_answer, family_keys)
    return IntentDecision(
        product_family=product_family,
        candidate_families=candidate_families,
        mentions_family=mentions_family,
        question_types=pick_question_types(
            response.answers[QUESTION_TYPE_QUESTION_ID], slot_model.question_type_margin
        ),
        family_probabilities=family_probabilities,
    )


async def decide_question_type_reply(
    client: DecisionClient, slot_model: SlotModel, query: str, reply: str
) -> str:
    response = await _decide(
        client,
        slot_model,
        {"question": query, "reply": reply},
        {QUESTION_TYPE_QUESTION_ID: question_type_question(_QUESTION_TYPE_REPLY_INSTRUCTIONS)},
    )
    question_type = response.answers[QUESTION_TYPE_QUESTION_ID].choice
    # The client already checked it's one of the options; this only guards
    # QUESTION_TYPES and QUESTION_TYPE_NAMES drifting apart.
    if question_type not in QUESTION_TYPE_NAMES:
        raise DecisionModelError(f"{question_type!r} isn't a question type")
    return question_type


async def decide_or_fall_back(
    slot_model: SlotModel, decide: Callable[[DecisionClient], Awaitable[ResultType]]
) -> ResultType | None:
    """Runs `decide` on the decision provider, or returns None after
    logging `slot_output_fallback` when the call fails for any reason -- the
    caller then runs its default chat path, like an assigned chat model's
    unusable output does. A decision provider has no OpenAI-direct twin, so
    this is also its outage fallback.
    """
    try:
        return await decide(get_decision_client())
    except DecisionModelError as error:
        log_slot_output_fallback(slot_model, f"{type(error).__name__}: {str(error)[:200]}")
        return None

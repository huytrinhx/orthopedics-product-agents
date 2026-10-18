"""The live compatibility probe (issue #24): the second layer of the check,
after the catalog filter. On save and from Test, it runs the slot's real
LangChain call shape against the chosen model with fixed sample inputs, and
a save is rejected with the reason when any check fails:

- structured-output slots call `with_structured_output(<the node's own
  schema>)`, and the result must parse into that schema. Intent
  classification runs both of its schemas: detect_intent's and the
  question-type reply classifier's;
- the ReAct agent calls `bind_tools(react_agent._TOOLS)` and must return a
  parsed tool call, then answers a second round replaying that call and a
  canned result, the way react_agent's loop does;
- the deterministic answer streams, and the result must have non-empty
  `.text`;
- tray-layout extraction runs the structured-output check with an image
  fixture kept beside this module (probe_fixtures/tray_layout.png);
- a decision model (issue #23, intent classification only) gets its own
  checks instead: detect_intent's family and question-type Choices must
  each come back as one of their own option keys, with probabilities, and
  so must the question-type reply's.

The checks are also backend/evals/openrouter_spike.py's, which runs them
under both APIs. No database, graph or tool execution: tool results are
canned.
"""
import asyncio
import base64
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from agents.intent_decisions import (
    FAMILY_QUESTION_ID,
    QUESTION_TYPE_QUESTION_ID,
    decide_question_type_reply,
    family_option_keys,
    family_question,
    question_type_question,
)
from agents.judge import _JudgeScores
from agents.workflows.deterministic import _IntentClassification, _QuestionTypeReply
from agents.workflows.react_agent import _AGENT_SYSTEM_PROMPT, _TOOLS
from config.llm_clients import DecisionClient, get_decision_client, openrouter_chat_model
from ingestion.entity_extraction import _ExtractionOutput
from ingestion.tray_layout_extraction import _SYSTEM_PROMPT as _TRAY_SYSTEM_PROMPT
from ingestion.tray_layout_extraction import _TrayLayoutExtraction
from model_assignments.runtime import SlotModel, decision_slot_model
from model_assignments.slots import ModelKind, Slot

# Generous: a reasoning model reading the tray image can take a while, but
# an admin is waiting on the Save button.
PROBE_TIMEOUT_SECONDS = 120.0

TRAY_IMAGE_FIXTURE = Path(__file__).parent / "probe_fixtures" / "tray_layout.png"

_KNOWN_PARTS = [
    {"sku": "MPPA100L", "description": "3.5mm x 20mm cannulated screw, full thread"},
    {"sku": "MPPA102L", "description": "3.5mm x 22mm cannulated screw, partial thread"},
    {"sku": "MGW1400", "description": "1.4mm guidewire, trocar tip"},
]
_KNOWN_TRAY = "MIS Foot Recon Implant Tray"
_CATALOG_LISTING = "\n".join(f"- {part['sku']}: {part['description']}" for part in _KNOWN_PARTS)

Check = Callable[[Any], Awaitable[str]]


class ProbeFailure(AssertionError):
    pass


def _require_schema(result: object, schema: type[BaseModel]) -> Any:
    if not isinstance(result, schema):
        raise ProbeFailure(f"output didn't parse into {schema.__name__}: got {type(result).__name__}")
    return result


async def check_detect_intent(model: Any) -> str:
    result = await model.with_structured_output(_IntentClassification).ainvoke(
        [
            SystemMessage(
                content=(
                    "Classify the sales rep's question. Allowed product families: "
                    "MIS - Foot Recon System, REFLEX HYBRID."
                )
            ),
            HumanMessage(content="What thread type is the 3.5mm MIS screw?"),
        ]
    )
    result = _require_schema(result, _IntentClassification)
    return f"family={result.product_family!r} types={result.question_type}"


async def check_question_type_reply(model: Any) -> str:
    result = await model.with_structured_output(_QuestionTypeReply).ainvoke(
        [
            SystemMessage(content="Pick the one question type the rep's reply means."),
            {"role": "user", "content": "Question: bunion steps?\nWhat they meant: I need the steps"},
        ]
    )
    return f"question_type={_require_schema(result, _QuestionTypeReply).question_type}"


async def check_judge(model: Any) -> str:
    result = await model.with_structured_output(_JudgeScores).ainvoke(
        [
            {"role": "system", "content": "Score the answer from 0 to 1 on each axis."},
            {
                "role": "user",
                "content": (
                    "Question: How long is MPPA100L?\n\n"
                    "Context given to the assistant:\n[doc-1#0] MPPA100L is 3.5mm x 20mm.\n\n"
                    "Assistant's answer:\nMPPA100L is 20mm long [doc-1#0]."
                ),
            },
        ]
    )
    return f"faithfulness={_require_schema(result, _JudgeScores).faithfulness}"


async def check_entity_extraction(model: Any) -> str:
    result = await model.with_structured_output(_ExtractionOutput).ainvoke(
        [
            ("system", "Extract part differentiations, referencing only known SKUs."),
            (
                "user",
                (
                    f"Known parts:\n{_CATALOG_LISTING}\n\nKnown trays:\n- {_KNOWN_TRAY}\n\n"
                    "Text:\nMPPA100L is fully threaded for compression across the joint, "
                    "while MPPA102L is partially threaded."
                ),
            ),
        ]
    )
    return f"differentiations={len(_require_schema(result, _ExtractionOutput).differentiations)}"


def tray_image_b64() -> str:
    """A stand-in tray photo: two labeled regions with SKUs printed in them,
    enough for the model to have something to read.
    """
    return base64.b64encode(TRAY_IMAGE_FIXTURE.read_bytes()).decode("ascii")


async def check_tray_layout(model: Any) -> str:
    result = await model.with_structured_output(_TrayLayoutExtraction).ainvoke(
        [
            ("system", _TRAY_SYSTEM_PROMPT),
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": (
                            f"Known trays:\n- {_KNOWN_TRAY}\n\nKnown parts:\n{_CATALOG_LISTING}\n\n"
                            "This page's own text:\nTray Layout - Top Level"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{tray_image_b64()}"},
                    },
                ],
            ),
        ]
    )
    return f"groups={len(_require_schema(result, _TrayLayoutExtraction).groups)}"


async def check_react_agent_tools(model: Any) -> str:
    bound = model.bind_tools(_TOOLS)
    exchange = [
        SystemMessage(content=_AGENT_SYSTEM_PROMPT),
        HumanMessage(content="What thread type is SKU MPPA100L? Look it up, don't guess."),
    ]
    first_response = await bound.ainvoke(exchange)
    if not first_response.tool_calls:
        raise ProbeFailure(f"no tool call on round 1: {first_response.text[:80]!r}")
    tool_messages = [
        ToolMessage(
            content=json.dumps([{"sku": "MPPA100L", "thread": "Full thread"}]),
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
        for tool_call in first_response.tool_calls
    ]
    # Round 2 replays round 1's response, as react_agent's scratchpad does.
    # That replay is where a stateless gateway would reject item ids from
    # the earlier response.
    second_response = await bound.ainvoke([*exchange, first_response, *tool_messages])
    called = ",".join(tool_call["name"] for tool_call in first_response.tool_calls)
    outcome = "more tools" if second_response.tool_calls else "answered"
    return f"called={called}; round 2 {outcome}"


async def check_streamed_generate(model: Any) -> str:
    chunk_count = 0
    text_parts: list[str] = []
    async for chunk in model.astream(
        [
            SystemMessage(content="You are OrthoMate. Answer in two short sentences."),
            HumanMessage(content="What is a cannulated screw used for?"),
        ]
    ):
        chunk_count += 1
        text_parts.append(chunk.text)
    answer = "".join(text_parts)
    if not answer.strip():
        raise ProbeFailure(f"streamed {chunk_count} chunk(s) with no text")
    return f"{chunk_count} chunks"


# Each slot's checks, by the name a failure is reported under.
SLOT_CHECKS: dict[Slot, dict[str, Check]] = {
    Slot.INTENT_CLASSIFICATION: {
        "detect_intent (structured)": check_detect_intent,
        "question-type reply (structured)": check_question_type_reply,
    },
    Slot.DETERMINISTIC_ANSWER: {"generate (streamed text)": check_streamed_generate},
    Slot.REACT_AGENT: {"react_agent (bind_tools, 2 rounds)": check_react_agent_tools},
    Slot.JUDGE: {"judge (structured)": check_judge},
    Slot.ENTITY_EXTRACTION: {"entity extraction (structured)": check_entity_extraction},
    Slot.TRAY_LAYOUT_EXTRACTION: {"tray layout (structured + image)": check_tray_layout},
}


_PROBE_FAMILIES = ["MIS - Foot Recon System", "REFLEX HYBRID"]


def _probe_decision_slot_model(model_id: str) -> SlotModel:
    return decision_slot_model(Slot.INTENT_CLASSIFICATION, model_id, "probe")


async def check_family_choice(model: tuple[DecisionClient, str]) -> str:
    """detect_intent's real request, through intent_decisions: the family
    Choice has to come back as one of its own keys (the client raises
    otherwise) with probabilities, and so does the question type's.
    """
    client, model_id = model
    family_keys = family_option_keys(_PROBE_FAMILIES)
    response = await client.decide(
        model_id,
        {"question": "What thread type is the 3.5mm MIS screw?"},
        {
            FAMILY_QUESTION_ID: family_question(family_keys, follow_up=False),
            QUESTION_TYPE_QUESTION_ID: question_type_question(),
        },
    )
    family_answer = response.answers[FAMILY_QUESTION_ID]
    if not family_answer.probabilities:
        raise ProbeFailure("the family choice came back with no probabilities")
    return (
        f"family={family_answer.choice!r} "
        f"p={family_answer.probabilities.get(family_answer.choice, 0.0):.2f} "
        f"type={response.answers[QUESTION_TYPE_QUESTION_ID].choice}; answered by {response.model}"
    )


async def check_question_type_reply_choice(model: tuple[DecisionClient, str]) -> str:
    client, model_id = model
    question_type = await decide_question_type_reply(
        client, _probe_decision_slot_model(model_id), "bunion steps?", "I need the steps"
    )
    return f"question_type={question_type}"


# A decision model's checks (issue #23), for the slots that accept one.
# Each check gets (client, model_id) instead of a chat model.
DECISION_SLOT_CHECKS: dict[Slot, dict[str, Check]] = {
    Slot.INTENT_CLASSIFICATION: {
        "detect_intent (family + question type choices)": check_family_choice,
        "question-type reply (choice)": check_question_type_reply_choice,
    },
}


@dataclass(frozen=True)
class ProbeResult:
    passed: bool
    # One line per check: what it saw, or why it failed.
    details: list[str]


async def run_check(check: Check, model: Any) -> tuple[bool, str]:
    started = time.monotonic()
    try:
        detail = await check(model)
    except Exception as error:  # noqa: BLE001 - every failure is a result to report
        return False, f"{type(error).__name__}: {str(error)[:300]}"
    return True, f"{detail} ({time.monotonic() - started:.1f}s)"


def build_probe_model(model_id: str, reasoning_effort: str | None) -> Any:
    """The gateway client alone, without get_chat_model()'s OpenAI-direct
    fallback: a probe that quietly passed on `CHAT_MODEL` would prove
    nothing about the model being saved. Tests replace this with a fake.
    """
    return openrouter_chat_model(model_id, reasoning_effort=reasoning_effort)


def build_probe_decision_client() -> DecisionClient:
    """The decision provider, which has no fallback to hide behind. Tests
    replace this with a fake.
    """
    return get_decision_client()


async def probe_slot(
    slot: Slot,
    model_id: str,
    reasoning_effort: str | None,
    model_kind: str = ModelKind.CHAT.value,
) -> ProbeResult:
    if model_kind == ModelKind.DECISION.value:
        model: Any = (build_probe_decision_client(), model_id)
        checks = DECISION_SLOT_CHECKS[slot]
    else:
        model = build_probe_model(model_id, reasoning_effort)
        checks = SLOT_CHECKS[slot]
    details: list[str] = []
    passed = True
    for check_name, check in checks.items():
        try:
            check_passed, detail = await asyncio.wait_for(
                run_check(check, model), timeout=PROBE_TIMEOUT_SECONDS
            )
        except TimeoutError:
            check_passed, detail = False, f"timed out after {PROBE_TIMEOUT_SECONDS:.0f}s"
        passed = passed and check_passed
        details.append(f"{'PASS' if check_passed else 'FAIL'} {check_name}: {detail}")
    return ProbeResult(passed, details)

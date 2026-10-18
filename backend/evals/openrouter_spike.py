"""Issue #8's spike: does every chat node's call shape work through
OpenRouter, under the Responses API and under Chat Completions? Its result
decides config/llm_clients.py's USE_RESPONSES_API and gets recorded in
docs/adr/0007-openrouter-chat-gateway.md.

Each check sends one node's real request shape to OpenRouter, built by the
same `openrouter_chat_model()` production uses, with only the API switched:

- with_structured_output with each node's own pydantic schema: detect_intent,
  the question-type reply classifier, the judge, entity extraction, and
  tray-layout extraction (with a rendered tray image as input);
- bind_tools with react_agent's real tools, then a second round replaying the
  tool call and its result, the way react_agent's loop does. That replay is
  where a stateless gateway would reject item ids from the earlier response;
- streamed text, the way deterministic `generate` streams through LangGraph.

No database, graph or tool execution: the tool result is canned. Needs only
OPENROUTER_API_KEY, and CHAT_MODEL for anything but the default model.

Usage:
    python -m evals.openrouter_spike
    python -m evals.openrouter_spike --reasoning-effort low
    python -m evals.openrouter_spike --api chat_completions
"""
import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from PIL import Image, ImageDraw

from agents.judge import _JudgeScores
from agents.workflows.deterministic import _IntentClassification, _QuestionTypeReply
from agents.workflows.react_agent import _AGENT_SYSTEM_PROMPT, _TOOLS
from config.llm_clients import configured_chat_model, openrouter_chat_model
from ingestion.entity_extraction import _ExtractionOutput
from ingestion.tray_layout_extraction import _SYSTEM_PROMPT as _TRAY_SYSTEM_PROMPT
from ingestion.tray_layout_extraction import _TrayLayoutExtraction

APIS = {"responses": True, "chat_completions": False}

_KNOWN_PARTS = [
    {"sku": "MPPA100L", "description": "3.5mm x 20mm cannulated screw, full thread"},
    {"sku": "MPPA102L", "description": "3.5mm x 22mm cannulated screw, partial thread"},
    {"sku": "MGW1400", "description": "1.4mm guidewire, trocar tip"},
]
_KNOWN_TRAY = "MIS Foot Recon Implant Tray"


async def _check_detect_intent(model: ChatOpenAI) -> str:
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
    return f"family={result.product_family!r} types={result.question_type}"


async def _check_question_type_reply(model: ChatOpenAI) -> str:
    result = await model.with_structured_output(_QuestionTypeReply).ainvoke(
        [
            SystemMessage(content="Pick the one question type the rep's reply means."),
            {"role": "user", "content": "Question: bunion steps?\nWhat they meant: I need the steps"},
        ]
    )
    return f"question_type={result.question_type}"


async def _check_judge(model: ChatOpenAI) -> str:
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
    return f"faithfulness={result.faithfulness}"


async def _check_entity_extraction(model: ChatOpenAI) -> str:
    catalog_listing = "\n".join(f"- {part['sku']}: {part['description']}" for part in _KNOWN_PARTS)
    result = await model.with_structured_output(_ExtractionOutput).ainvoke(
        [
            ("system", "Extract part differentiations, referencing only known SKUs."),
            (
                "user",
                (
                    f"Known parts:\n{catalog_listing}\n\nKnown trays:\n- {_KNOWN_TRAY}\n\n"
                    "Text:\nMPPA100L is fully threaded for compression across the joint, "
                    "while MPPA102L is partially threaded."
                ),
            ),
        ]
    )
    return f"differentiations={len(result.differentiations)}"


def _tray_image_b64() -> str:
    """A stand-in tray photo: two labeled regions with SKUs printed in
    them, enough for the model to have something to read.
    """
    image = Image.new("RGB", (640, 320), "white")
    drawing = ImageDraw.Draw(image)
    drawing.text((20, 10), "TOP LEVEL", fill="black")
    drawing.rectangle((20, 40, 300, 300), outline="black", width=3)
    drawing.text((40, 60), "LEFT SIDE", fill="black")
    drawing.text((40, 120), "MPPA100L  3.5 x 20 FT", fill="black")
    drawing.rectangle((340, 40, 620, 300), outline="black", width=3)
    drawing.text((360, 60), "RIGHT SIDE", fill="black")
    drawing.text((360, 120), "MGW1400  1.4mm WIRE", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def _check_tray_layout(model: ChatOpenAI) -> str:
    catalog_listing = "\n".join(f"- {part['sku']}: {part['description']}" for part in _KNOWN_PARTS)
    result = await model.with_structured_output(_TrayLayoutExtraction).ainvoke(
        [
            ("system", _TRAY_SYSTEM_PROMPT),
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": (
                            f"Known trays:\n- {_KNOWN_TRAY}\n\nKnown parts:\n{catalog_listing}\n\n"
                            "This page's own text:\nTray Layout - Top Level"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_tray_image_b64()}"},
                    },
                ],
            ),
        ]
    )
    return f"groups={len(result.groups)}"


async def _check_react_agent_tools(model: ChatOpenAI) -> str:
    bound = model.bind_tools(_TOOLS)
    exchange = [
        SystemMessage(content=_AGENT_SYSTEM_PROMPT),
        HumanMessage(content="What thread type is SKU MPPA100L? Look it up, don't guess."),
    ]
    first_response = await bound.ainvoke(exchange)
    if not first_response.tool_calls:
        raise AssertionError(f"no tool call on round 1: {first_response.text[:80]!r}")
    tool_messages = [
        ToolMessage(
            content=json.dumps([{"sku": "MPPA100L", "thread": "Full thread"}]),
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
        for tool_call in first_response.tool_calls
    ]
    # Round 2 replays round 1's response, as react_agent's scratchpad does.
    second_response = await bound.ainvoke([*exchange, first_response, *tool_messages])
    called = ",".join(tool_call["name"] for tool_call in first_response.tool_calls)
    outcome = "more tools" if second_response.tool_calls else "answered"
    return f"called={called}; round 2 {outcome}"


async def _check_streamed_generate(model: ChatOpenAI) -> str:
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
    if chunk_count < 2 or not answer.strip():
        raise AssertionError(f"{chunk_count} chunk(s), text={answer[:80]!r}")
    return f"{chunk_count} chunks"


CHECKS: dict[str, Callable[[ChatOpenAI], Awaitable[str]]] = {
    "detect_intent (structured)": _check_detect_intent,
    "question-type reply (structured)": _check_question_type_reply,
    "judge (structured)": _check_judge,
    "entity extraction (structured)": _check_entity_extraction,
    "tray layout (structured + image)": _check_tray_layout,
    "react_agent (bind_tools, 2 rounds)": _check_react_agent_tools,
    "generate (streamed text)": _check_streamed_generate,
}


async def _run_check(
    check: Callable[[ChatOpenAI], Awaitable[str]], model: ChatOpenAI
) -> tuple[bool, str]:
    started = time.monotonic()
    try:
        detail = await check(model)
    except Exception as error:  # noqa: BLE001 - every failure is a result to report
        return False, f"{type(error).__name__}: {str(error)[:300]}"
    return True, f"{detail} ({time.monotonic() - started:.1f}s)"


async def run_spike(model_id: str, api_names: list[str], reasoning_effort: str | None) -> bool:
    results: dict[str, dict[str, tuple[bool, str]]] = {name: {} for name in CHECKS}
    for api_name in api_names:
        model = openrouter_chat_model(
            model_id, reasoning_effort=reasoning_effort, use_responses_api=APIS[api_name]
        )
        for check_name, check in CHECKS.items():
            results[check_name][api_name] = await _run_check(check, model)

    print(f"model={model_id} reasoning_effort={reasoning_effort or '(model default)'}\n")
    name_width = max(len(name) for name in CHECKS)
    print(" | ".join([f"{'check':<{name_width}}", *(f"{api_name:<16}" for api_name in api_names)]))
    print("-" * (name_width + 19 * len(api_names)))
    for check_name, by_api in results.items():
        cells = [f"{'PASS' if by_api[api_name][0] else 'FAIL':<16}" for api_name in api_names]
        print(" | ".join([f"{check_name:<{name_width}}", *cells]))

    print("\nDetails:")
    for check_name, by_api in results.items():
        for api_name, (passed, detail) in by_api.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {api_name} / {check_name}: {detail}")
    return all(passed for by_api in results.values() for passed, _ in by_api.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every chat node's call shape through OpenRouter under both APIs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", help="OpenRouter model id (defaults to CHAT_MODEL, e.g. openai/gpt-5.6-luna)"
    )
    parser.add_argument("--api", choices=sorted(APIS), action="append", help="repeatable; default both")
    parser.add_argument("--reasoning-effort", help="e.g. low/medium/high; default is the model's own")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY isn't set -- this spike only talks to OpenRouter.")
    all_passed = asyncio.run(
        run_spike(args.model or configured_chat_model(), args.api or list(APIS), args.reasoning_effort)
    )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

"""OpenRouter's live model catalog, for the Settings picker: the first,
static layer of the compatibility check. A model is offered for a slot only
when its catalog metadata covers the slot's needs (slots.py) --
`supported_parameters` for tools and structured outputs,
`architecture.input_modalities` for image input. The live probe (probe.py)
is the second layer.

`fetch_openrouter_models` is the one network call, and tests replace it.
The parsed catalog is cached in-process for an hour.

Decision models (issue #23) are a separate listing,
`fetch_openrouter_decision_models`, offered only to a slot that accepts
them, by their dated canonical slug. They have no needs to filter on:
their probe (probe.py) is the whole check.
"""
import re
import time
from dataclasses import dataclass, replace

import httpx

from config.llm_clients import OPENROUTER_BASE_URL
from model_assignments.slots import SLOT_DEFINITIONS, Need, Slot

CATALOG_CACHE_SECONDS = 3600.0
_CATALOG_TIMEOUT_SECONDS = 20.0

# A trailing release date on a model id: -2024-11-20, -20240620, or a
# four-digit -2411/-0613 stamp.
_DATED_SUFFIX = r"-(\d{4}-\d{2}-\d{2}|\d{8}|\d{4})"

_cached_models: list["CatalogModel"] | None = None
_cached_at = 0.0
_cached_decision_models: list["CatalogModel"] | None = None
_decision_cached_at = 0.0


class CatalogUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogModel:
    model_id: str
    name: str
    # Dollars per million tokens, None when OpenRouter doesn't list a fixed
    # price (a router like openrouter/auto).
    prompt_price_per_million: float | None
    completion_price_per_million: float | None
    supported_parameters: frozenset[str]
    input_modalities: frozenset[str]
    output_modalities: frozenset[str] = frozenset()
    # A decision model's other listed id (its `id`, when that isn't the
    # dated slot it's offered under), so a hand-typed alias still finds it.
    listed_id: str | None = None

    @property
    def supports_reasoning(self) -> bool:
        return "reasoning" in self.supported_parameters

    @property
    def is_decision_model(self) -> bool:
        """Answers OpenRouter's Decisions API, not chat (issue #23)."""
        return "decisions" in self.output_modalities

    def covers(self, need: Need) -> bool:
        if need is Need.STRUCTURED_OUTPUT:
            return "structured_outputs" in self.supported_parameters
        if need is Need.TOOL_CALLING:
            return "tools" in self.supported_parameters
        if need is Need.IMAGE_INPUT:
            return "image" in self.input_modalities
        # Every chat model streams text.
        return True

    def missing_needs(self, slot: Slot) -> list[Need]:
        return [need for need in SLOT_DEFINITIONS[slot].needs if not self.covers(need)]


async def fetch_openrouter_models() -> list[dict]:
    """GET /models: the raw `data` list. Public, so no key is sent."""
    try:
        async with httpx.AsyncClient(timeout=_CATALOG_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{OPENROUTER_BASE_URL}/models")
            response.raise_for_status()
            return response.json()["data"]
    except (httpx.HTTPError, KeyError, ValueError) as error:
        raise CatalogUnavailable(f"OpenRouter's model catalog couldn't be loaded: {error}") from error


def _price_per_million(price: object) -> float | None:
    try:
        per_token = float(price)
    except (TypeError, ValueError):
        return None
    return None if per_token < 0 else round(per_token * 1_000_000, 4)


async def fetch_openrouter_decision_models() -> list[dict]:
    """GET /models?output_modalities=decisions: the raw `data` list of
    decision models (issue #23). Public, like the chat catalog.
    """
    try:
        async with httpx.AsyncClient(timeout=_CATALOG_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f"{OPENROUTER_BASE_URL}/models", params={"output_modalities": "decisions"}
            )
            response.raise_for_status()
            return response.json()["data"]
    except (httpx.HTTPError, KeyError, ValueError) as error:
        raise CatalogUnavailable(
            f"OpenRouter's decision model catalog couldn't be loaded: {error}"
        ) from error


def _parse(raw_model: dict, model_id: str | None = None) -> CatalogModel:
    pricing = raw_model.get("pricing") or {}
    architecture = raw_model.get("architecture") or {}
    return CatalogModel(
        model_id=model_id or raw_model["id"],
        name=raw_model.get("name") or raw_model["id"],
        prompt_price_per_million=_price_per_million(pricing.get("prompt")),
        completion_price_per_million=_price_per_million(pricing.get("completion")),
        supported_parameters=frozenset(raw_model.get("supported_parameters") or []),
        input_modalities=frozenset(architecture.get("input_modalities") or []),
        output_modalities=frozenset(architecture.get("output_modalities") or []),
    )


def _parse_decision_model(raw_model: dict) -> CatalogModel | None:
    """A decision model under its dated `canonical_slug` -- the exact build,
    which requests accept and responses name -- rather than its `id`, which
    may be an alias that moves (a leading `~` always is). One with neither
    a canonical slug nor a non-alias id isn't offered.
    """
    canonical_slug = raw_model.get("canonical_slug")
    model_id = raw_model.get("id") or ""
    pinned_id = canonical_slug or (model_id if not model_id.startswith("~") else None)
    if not pinned_id:
        return None
    return replace(_parse(raw_model, pinned_id), listed_id=model_id or None)


def prefer_dated_ids(models: list[CatalogModel]) -> list[CatalogModel]:
    """Drops an alias like `openai/gpt-4o` when the catalog also lists a
    dated id for it (`openai/gpt-4o-2024-11-20`): a dated id keeps meaning
    the same model, while an alias can move to a newer one underneath a
    saved setting. Aliases with no dated sibling stay.
    """
    model_ids = {model.model_id for model in models}
    dated_bases = {
        match.group(1)
        for model_id in model_ids
        if (match := re.fullmatch(r"(.+)" + _DATED_SUFFIX, model_id))
    }
    aliases_with_dated_ids = dated_bases & model_ids
    return [model for model in models if model.model_id not in aliases_with_dated_ids]


async def list_catalog_models() -> list[CatalogModel]:
    global _cached_models, _cached_at
    if _cached_models is None or time.monotonic() - _cached_at >= CATALOG_CACHE_SECONDS:
        raw_models = await fetch_openrouter_models()
        _cached_models = sorted(
            (_parse(raw_model) for raw_model in raw_models if raw_model.get("id")),
            key=lambda model: model.name.lower(),
        )
        _cached_at = time.monotonic()
    return _cached_models


async def list_decision_models() -> list[CatalogModel]:
    global _cached_decision_models, _decision_cached_at
    if (
        _cached_decision_models is None
        or time.monotonic() - _decision_cached_at >= CATALOG_CACHE_SECONDS
    ):
        raw_models = await fetch_openrouter_decision_models()
        parsed = {
            model.model_id: model
            for model in (_parse_decision_model(raw_model) for raw_model in raw_models)
            if model is not None
        }
        _cached_decision_models = sorted(parsed.values(), key=lambda model: model.name.lower())
        _decision_cached_at = time.monotonic()
    return _cached_decision_models


def clear_catalog_cache() -> None:
    global _cached_models, _cached_decision_models
    _cached_models = None
    _cached_decision_models = None


async def models_for_slot(slot: Slot) -> list[CatalogModel]:
    """Chat models only: a decision model can't answer a chat call, even
    where its metadata would pass the slot's needs.
    """
    return [
        model
        for model in prefer_dated_ids(await list_catalog_models())
        if not model.is_decision_model and not model.missing_needs(slot)
    ]


async def decision_models_for_slot(slot: Slot) -> list[CatalogModel]:
    """The decision models a slot may run, by dated id; none for a slot
    that doesn't accept them.
    """
    if not SLOT_DEFINITIONS[slot].accepts_decision_models:
        return []
    return await list_decision_models()


async def find_decision_model(model_id: str) -> CatalogModel | None:
    """By its dated id or its listed one. Either way the result's
    `model_id` is the dated id, which is what gets saved or run.
    """
    return next(
        (
            model
            for model in await list_decision_models()
            if model_id in (model.model_id, model.listed_id)
        ),
        None,
    )


async def find_catalog_model(model_id: str) -> CatalogModel | None:
    """Any listed id, aliases included: the picker prefers dated ids, but a
    saved alias or a hand-typed override is still a real model.
    """
    return next((model for model in await list_catalog_models() if model.model_id == model_id), None)

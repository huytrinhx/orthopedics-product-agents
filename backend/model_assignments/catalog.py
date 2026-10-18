"""OpenRouter's live model catalog, for the Settings picker: the first,
static layer of the compatibility check. A model is offered for a slot only
when its catalog metadata covers the slot's needs (slots.py) --
`supported_parameters` for tools and structured outputs,
`architecture.input_modalities` for image input. The live probe (probe.py)
is the second layer.

`fetch_openrouter_models` is the one network call, and tests replace it.
The parsed catalog is cached in-process for an hour.
"""
import re
import time
from dataclasses import dataclass

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

    @property
    def supports_reasoning(self) -> bool:
        return "reasoning" in self.supported_parameters

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


def _parse(raw_model: dict) -> CatalogModel:
    pricing = raw_model.get("pricing") or {}
    architecture = raw_model.get("architecture") or {}
    return CatalogModel(
        model_id=raw_model["id"],
        name=raw_model.get("name") or raw_model["id"],
        prompt_price_per_million=_price_per_million(pricing.get("prompt")),
        completion_price_per_million=_price_per_million(pricing.get("completion")),
        supported_parameters=frozenset(raw_model.get("supported_parameters") or []),
        input_modalities=frozenset(architecture.get("input_modalities") or []),
    )


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


def clear_catalog_cache() -> None:
    global _cached_models
    _cached_models = None


async def models_for_slot(slot: Slot) -> list[CatalogModel]:
    return [model for model in prefer_dated_ids(await list_catalog_models()) if not model.missing_needs(slot)]


async def find_catalog_model(model_id: str) -> CatalogModel | None:
    """Any listed id, aliases included: the picker prefers dated ids, but a
    saved alias or a hand-typed override is still a real model.
    """
    return next((model for model in await list_catalog_models() if model.model_id == model_id), None)

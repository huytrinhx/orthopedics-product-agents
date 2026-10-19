"""Resolves a step's items to SKU sets in code (issue #28), never in the
prompt.

A step names an instrument and a spec ("Ø2.6mm drill bit for Ø3.7mm
screws"); the SKUs sit in tray tables pages later. Each item resolves
against the catalog parts in the document's scope:

1. A SKU the text literally printed, if it's in the catalog (`literal`).
2. Otherwise the parts whose kind matches the item's type. A catalog
   description leads with its noun ("DRILL BIT, CANN, AO/QC, 2.6MM, ..."),
   so the kind is read from there; a screw is a part whose noun says SCREW
   or that has a driver spec (MIS screws are "MIS HV CHAMFER FT, 4.0 X 26").
3. Narrowed by the item's specs:
   - `diameter` is the part's own dimension: the number before " X ", else
     the first number with "MM", else the first number. "2.6MM, 3.7 PILOT"
     is a 2.6mm drill, not a 3.7mm one.
   - `thread` (FT/PT) and `side` (L/R) drop parts that say otherwise and
     keep the ones that say nothing.
   - `driver` needs its token (T15) in the description.
   - `for_screw_diameter` on a drill, guidewire or driver goes through the
     screws: the in-scope screws of that diameter (and `for_screw_thread`)
     give their `pre_drill_spec`, `guidewire_spec` or `driver_spec`, which
     the tool must match. Without such screws, or on any other kind (a
     countersink, a tissue protector), the screw's diameter has to be in
     the tool's description ("COUNTERSINK, ..., 4.0 MIS FT").
   - A drill, guidewire or driver with neither its own diameter nor a
     screw spec takes the screws its own step resolved ("the appropriate
     driver" under the screw step).
   - Any other spec with numbers needs them all in the description ("1.6 x
     200mm"); alternatives ("20, 30, or 40 mm") need one; a range ("20 -
     50mm") needs a number inside it. A spec with no number narrows only
     when something is left.
4. Of what's left, the SKUs the document's own SKU index found
   (`in_document`). When none of them is in the document, they're all kept
   (`catalog_only`), so a reviewer sees the candidates; nothing left is
   `unresolved`.

The document's scope (`scope_families`) is its own family, the families of
every catalog SKU its index found, and the families of the trays its
techniques say they require. The index is the evidence that a document
uses another family's parts: REFLEX HYBRID items in the REFLEX document,
Distal Tibia items in Ankle Trauma.
"""
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ingestion.technique_extraction import StepItemReference

LITERAL = "literal"
IN_DOCUMENT = "in_document"
CATALOG_ONLY = "catalog_only"
UNRESOLVED = "unresolved"

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
_DIMENSION_BY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[xX×]\s*\d")
_MILLIMETRE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*MM\b", re.IGNORECASE)
_RANGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm\s*)?(?:-|–|—|to)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_ALTERNATIVE_SEPARATOR_PATTERN = re.compile(r"\s*(?:,|/|;|\bor\b|\band\b)\s*", re.IGNORECASE)
_DRIVER_TOKEN_PATTERN = re.compile(r"\bT\d{1,2}\b", re.IGNORECASE)
_WORD_PATTERN = re.compile(r"[A-Za-z]{2,}")
_TRADEMARK_PATTERN = re.compile(r"[®™©]")

# The kinds of tool a screw's catalog row names a spec for.
DRILL = "drill"
GUIDEWIRE = "guidewire"
DRIVER = "driver"
_TOOL_SPEC_COLUMN = {DRILL: "pre_drill_spec", GUIDEWIRE: "guidewire_spec", DRIVER: "driver_spec"}


@dataclass(frozen=True)
class _Kind:
    # The description nouns (the text before its first comma) of this kind.
    nouns: tuple[str, ...]
    # Words the description must also carry.
    required_words: tuple[str, ...] = ()
    tool: str | None = None
    screw: bool = False


_SCREW_KIND = _Kind(nouns=("SCREW",), screw=True)
_GUIDEWIRE_KIND = _Kind(
    nouns=("GUIDEWIRE", "GUIDEPIN", "GUIDE PIN", "GUIDE WIRE", "K-WIRE", "WIRE"), tool=GUIDEWIRE
)
_KINDS: dict[str, _Kind] = {
    "DRILL BIT": _Kind(nouns=("DRILL BIT", "DRILL"), tool=DRILL),
    "DRILL": _Kind(nouns=("DRILL BIT", "DRILL"), tool=DRILL),
    "GUIDEWIRE": _GUIDEWIRE_KIND,
    "GUIDE WIRE": _GUIDEWIRE_KIND,
    "GUIDEPIN": _GUIDEWIRE_KIND,
    "GUIDE PIN": _GUIDEWIRE_KIND,
    "K-WIRE": _GUIDEWIRE_KIND,
    "WIRE": _GUIDEWIRE_KIND,
    "DRIVER": _Kind(nouns=("DRIVER",), tool=DRIVER),
    "SCREW": _SCREW_KIND,
    "STAPLE": _Kind(nouns=("STAPLE",)),
    "GUIDE": _Kind(nouns=("GUIDE",)),
    "LOCATOR PIN": _Kind(nouns=("PIN",), required_words=("LOCATOR",)),
    "DRILL PIN": _Kind(nouns=("DRILL PIN",)),
}


@dataclass(frozen=True)
class CatalogPart:
    sku: str
    description: str
    families: frozenset[str] = frozenset()
    item_type: str | None = None
    thread: str | None = None
    guidewire_spec: str | None = None
    pre_drill_spec: str | None = None
    driver_spec: str | None = None

    @property
    def noun(self) -> str:
        return self.description.split(",")[0].strip().upper()

    @property
    def numbers(self) -> list[float]:
        return [float(number) for number in _NUMBER_PATTERN.findall(self.description)]

    @property
    def dimension(self) -> float | None:
        for pattern in (_DIMENSION_BY_PATTERN, _MILLIMETRE_PATTERN, _NUMBER_PATTERN):
            match = pattern.search(self.description)
            if match:
                return float(match.group(1) if pattern is not _NUMBER_PATTERN else match.group(0))
        return None

    @property
    def is_screw(self) -> bool:
        upper = self.description.upper()
        if "SCREW" in self.noun:
            return True
        return bool(self.driver_spec) and "STAPLE" not in upper and "PLATE" not in upper

    @property
    def thread_kind(self) -> str | None:
        """"FT" or "PT", from the Thread column or the description."""
        if self.thread:
            if self.thread.lower().startswith("full"):
                return "FT"
            if self.thread.lower().startswith("partial"):
                return "PT"
        words = set(re.findall(r"\b[A-Z]{2}\b", self.description.upper()))
        if "FT" in words:
            return "FT"
        if "PT" in words:
            return "PT"
        return None

    @property
    def side(self) -> str | None:
        upper = self.description.upper()
        if re.search(r"\bLEFT\b", upper):
            return "L"
        if re.search(r"\bRIGHT\b", upper):
            return "R"
        return None


@dataclass
class ResolutionContext:
    """What one document's items resolve against."""

    catalog: Mapping[str, CatalogPart]
    scope: frozenset[str]
    # Catalog SKU -> the chunks the document's SKU index found it in.
    document_sku_chunks: Mapping[str, list[int]] = field(default_factory=dict)

    def scoped_parts(self) -> list[CatalogPart]:
        return [
            part
            for part in self.catalog.values()
            if part.families & self.scope or part.sku in self.document_sku_chunks
        ]


@dataclass(frozen=True)
class ResolvedItem:
    skus: tuple[str, ...]
    resolution: str
    # The step's chunks, then the chunks the index places the SKUs in.
    source_chunk_indexes: tuple[int, ...]


def scope_families(
    document_family: str | None,
    indexed_skus: Iterable[str],
    catalog: Mapping[str, CatalogPart],
    required_tray_families: Iterable[str] = (),
) -> frozenset[str]:
    families: set[str] = {document_family} if document_family else set()
    for sku in indexed_skus:
        part = catalog.get(sku)
        if part is not None:
            families |= part.families
    families |= set(required_tray_families)
    return frozenset(families)


def _normalized_name(name: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _TRADEMARK_PATTERN.sub("", name).lower()))


def resolve_tray_references(
    references: Iterable[str], trays_by_family: Mapping[str, list[str]]
) -> list[str]:
    """The catalog tray names `references` mean, matched on name with case,
    punctuation and trademark signs ignored. A reference naming a family
    ("Core Foot Plating System") means all of its trays. Unmatched
    references are dropped."""
    trays_by_name = {
        _normalized_name(tray): tray for trays in trays_by_family.values() for tray in trays
    }
    families_by_name = {_normalized_name(family): family for family in trays_by_family}
    resolved: list[str] = []
    for reference in references:
        name = _normalized_name(reference)
        if name in trays_by_name:
            matched = [trays_by_name[name]]
        elif name in families_by_name:
            matched = list(trays_by_family[families_by_name[name]])
        else:
            matched = []
        resolved.extend(tray for tray in matched if tray not in resolved)
    return resolved


def normalize_parameter_name(name: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", name.lower()))


def normalize_parameter_value(value: str) -> str:
    """"Ø3.7mm", "3.70" and "3.7 mm screws" all become "3.7"; a value with
    no number is its lowercase words ("FT" -> "ft")."""
    numbers = _NUMBER_PATTERN.findall(value)
    if numbers:
        return "/".join(f"{float(number):g}" for number in numbers)
    return " ".join(word.lower() for word in _WORD_PATTERN.findall(value))


def applies_to(applies_when: Mapping[str, str], parameters: Mapping[str, str]) -> bool:
    """Whether an item bound to `applies_when` is used for the surgeon's
    `parameters`. A parameter the caller didn't give doesn't exclude
    anything: every option's item comes back for the caller to list or ask
    about."""
    given = {
        normalize_parameter_name(name): normalize_parameter_value(value)
        for name, value in parameters.items()
    }
    for name, value in applies_when.items():
        normalized_name = normalize_parameter_name(name)
        if normalized_name in given and given[normalized_name] != normalize_parameter_value(value):
            return False
    return True


def _kind_for(item_type: str) -> _Kind | None:
    normalized = re.sub(r"\s+", " ", item_type.strip().upper())
    if normalized in _KINDS:
        return _KINDS[normalized]
    if normalized.endswith("S") and normalized[:-1] in _KINDS:
        return _KINDS[normalized[:-1]]
    return None


def _matches_kind(part: CatalogPart, item_type: str) -> bool:
    kind = _kind_for(item_type)
    if kind is not None and kind.screw:
        return part.is_screw
    if kind is not None:
        noun = part.noun
        if not any(noun == candidate or noun.endswith(" " + candidate) for candidate in kind.nouns):
            return False
        upper = part.description.upper()
        return all(word in upper for word in kind.required_words)
    # A type with no entry ("countersink", "tissue protector", "tamp"):
    # the description's noun has to contain it.
    normalized = re.sub(r"\s+", " ", item_type.strip().upper())
    singular = normalized.removesuffix("S")
    return bool(singular) and (singular in part.noun or normalized in part.noun)


def _numbers(value: str) -> list[float]:
    return [float(number) for number in _NUMBER_PATTERN.findall(value)]


def _numeric_alternatives(value: str) -> list[list[float]]:
    return [
        numbers
        for alternative in _ALTERNATIVE_SEPARATOR_PATTERN.split(value)
        if (numbers := _numbers(alternative))
    ]


def _matches_numeric_spec(part: CatalogPart, value: str) -> bool:
    numbers = set(part.numbers)
    range_match = _RANGE_PATTERN.search(value)
    if range_match:
        low, high = sorted((float(range_match.group(1)), float(range_match.group(2))))
        return any(low <= number <= high for number in numbers)
    return any(set(alternative) <= numbers for alternative in _numeric_alternatives(value))


def _matches_dimension(part: CatalogPart, value: str) -> bool:
    dimension = part.dimension
    if dimension is None:
        return False
    return any(dimension in alternative for alternative in _numeric_alternatives(value))


def _wanted_thread(value: str) -> str | None:
    upper = value.upper()
    if re.search(r"\bFT\b|FULL", upper):
        return "FT"
    if re.search(r"\bPT\b|PARTIAL|HALF", upper):
        return "PT"
    return None


def _wanted_side(value: str) -> str | None:
    upper = value.strip().upper()
    if upper in {"L", "LEFT"} or upper.startswith("LEFT"):
        return "L"
    if upper in {"R", "RIGHT"} or upper.startswith("RIGHT"):
        return "R"
    return None


def _not_contradicting(parts: list[CatalogPart], attribute: str, wanted: str | None) -> list[CatalogPart]:
    if wanted is None:
        return parts
    return [part for part in parts if getattr(part, attribute) in (None, wanted)]


def _matches_tool_spec(part: CatalogPart, tool: str, spec: str) -> bool:
    if tool == DRIVER:
        tokens = {token.upper() for token in _DRIVER_TOKEN_PATTERN.findall(spec)}
        return bool(tokens) and tokens <= {
            token.upper() for token in _DRIVER_TOKEN_PATTERN.findall(part.description)
        }
    numbers = set(_numbers(spec))
    return bool(numbers) and numbers <= set(part.numbers)


def _screws_for(
    parts: list[CatalogPart], diameter: str | None, thread: str | None
) -> list[CatalogPart]:
    screws = [part for part in parts if part.is_screw]
    if diameter is not None:
        screws = [part for part in screws if _matches_dimension(part, diameter)]
    return _not_contradicting(screws, "thread_kind", _wanted_thread(thread) if thread else None)


def _through_screws(candidates: list[CatalogPart], tool: str, screws: list[CatalogPart]) -> list[CatalogPart]:
    column = _TOOL_SPEC_COLUMN[tool]
    specs = {getattr(screw, column) for screw in screws if getattr(screw, column)}
    return [
        part for part in candidates if any(_matches_tool_spec(part, tool, spec) for spec in specs)
    ]


def _narrow_by_specs(
    candidates: list[CatalogPart],
    item: StepItemReference,
    scoped_parts: list[CatalogPart],
    step_screws: list[CatalogPart],
) -> list[CatalogPart]:
    specs = {normalize_parameter_name(name): value for name, value in item.spec_values().items()}
    kind = _kind_for(item.item_type)
    tool = kind.tool if kind is not None else None

    for name, value in specs.items():
        if name in {"for_screw_diameter", "for_screw_thread"}:
            continue
        if name == "thread":
            candidates = _not_contradicting(candidates, "thread_kind", _wanted_thread(value))
        elif name == "side":
            candidates = _not_contradicting(candidates, "side", _wanted_side(value))
        elif name == "driver":
            if _DRIVER_TOKEN_PATTERN.search(value):
                candidates = [part for part in candidates if _matches_tool_spec(part, DRIVER, value)]
        elif name == "diameter":
            if _numbers(value):
                candidates = [part for part in candidates if _matches_dimension(part, value)]
        elif _numbers(value):
            candidates = [part for part in candidates if _matches_numeric_spec(part, value)]
        else:
            words = [word.upper() for word in _WORD_PATTERN.findall(value)]
            narrowed = [
                part for part in candidates if all(word in part.description.upper() for word in words)
            ]
            candidates = narrowed or candidates

    screw_diameter = specs.get("for_screw_diameter")
    screw_thread = specs.get("for_screw_thread")
    has_own_diameter = "diameter" in specs and bool(_numbers(specs["diameter"]))
    if tool is not None and not has_own_diameter and (screw_diameter or screw_thread):
        screws = _screws_for(scoped_parts, screw_diameter, screw_thread)
        through_screws = _through_screws(candidates, tool, screws) if screws else []
        if through_screws:
            candidates = through_screws
        elif screw_diameter:
            candidates = [part for part in candidates if _matches_numeric_spec(part, screw_diameter)]
    elif tool is not None and not has_own_diameter and step_screws and not specs.get("driver"):
        through_screws = _through_screws(candidates, tool, step_screws)
        candidates = through_screws or candidates
    elif screw_diameter and _numbers(screw_diameter):
        candidates = [part for part in candidates if _matches_numeric_spec(part, screw_diameter)]
        if screw_thread:
            candidates = _not_contradicting(candidates, "thread_kind", _wanted_thread(screw_thread))
    return candidates


def _source_chunks(
    step_chunk_indexes: Iterable[int], skus: Iterable[str], context: ResolutionContext
) -> tuple[int, ...]:
    chunks = list(dict.fromkeys(step_chunk_indexes))
    for sku in skus:
        for chunk_index in context.document_sku_chunks.get(sku, []):
            if chunk_index not in chunks:
                chunks.append(chunk_index)
    return tuple(chunks)


def resolve_item(
    item: StepItemReference,
    context: ResolutionContext,
    *,
    step_chunk_indexes: Iterable[int] = (),
    step_screws: Iterable[str] = (),
) -> ResolvedItem:
    """One item's SKU set. `step_screws` are the screws the item's own step
    resolved, for a tool the step names with no spec of its own."""
    step_chunk_indexes = list(step_chunk_indexes)
    literal = (item.sku or "").strip().upper()
    if literal and literal in context.catalog:
        return ResolvedItem((literal,), LITERAL, _source_chunks(step_chunk_indexes, [literal], context))

    scoped_parts = context.scoped_parts()
    candidates = [part for part in scoped_parts if _matches_kind(part, item.item_type)]
    candidates = _narrow_by_specs(
        candidates,
        item,
        scoped_parts,
        [context.catalog[sku] for sku in step_screws if sku in context.catalog],
    )
    in_document = sorted(part.sku for part in candidates if part.sku in context.document_sku_chunks)
    if in_document:
        skus, resolution = in_document, IN_DOCUMENT
    elif candidates:
        skus, resolution = sorted(part.sku for part in candidates), CATALOG_ONLY
    else:
        skus, resolution = [], UNRESOLVED
    return ResolvedItem(tuple(skus), resolution, _source_chunks(step_chunk_indexes, skus, context))


def resolve_step_items(
    items: list[StepItemReference], context: ResolutionContext, step_chunk_indexes: list[int]
) -> list[ResolvedItem]:
    """Every item of one step, in order. Screws first, so a tool the step
    names with no spec ("the appropriate driver") resolves through them."""
    resolved: dict[int, ResolvedItem] = {}
    screw_positions = [
        position
        for position, item in enumerate(items)
        if (kind := _kind_for(item.item_type)) is not None and kind.screw
    ]
    for position in screw_positions:
        resolved[position] = resolve_item(items[position], context, step_chunk_indexes=step_chunk_indexes)
    step_screws = [sku for position in screw_positions for sku in resolved[position].skus]
    for position, item in enumerate(items):
        if position not in resolved:
            resolved[position] = resolve_item(
                item, context, step_chunk_indexes=step_chunk_indexes, step_screws=step_screws
            )
    return [resolved[position] for position in range(len(items))]

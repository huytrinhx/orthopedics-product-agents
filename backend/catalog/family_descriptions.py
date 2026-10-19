"""A short description of each Product Family for the intent classifiers
(issue #30), generated from the catalog, never written by hand per family.

Both classifiers -- the chat path's "Allowed product families" list and a
decision model's family Choice (agents/intent_decisions.py) -- used to see
bare family names. That isn't enough to choose between them: nothing says
TETRA is a REFLEX staple, that "MIS" on its own means the Foot Recon
system rather than the power tools, or that "Cannulated Screws" isn't where
the screws in the MIS set live. Each description has up to three parts:

- The family's tray names (display names), minus any that only restate the
  family's own name -- most families have exactly one such tray.
- Up to MAX_DISTINCTIVE_TERMS distinctive terms from its Parts'
  descriptions. A catalog description is comma-separated segments
  (`STAPLE, NON-S, REFLEX ULTRA, 15X15`); each segment, with sizes, single
  letters, variant qualifiers (LEFT, NON-S) and the family's own name
  words dropped, is a candidate term (`ULTRA`). A term is distinctive when
  no other family has its words in one Part segment (HYBRID's LOCATOR
  isn't: REFLEX has STAPLE LOCATOR) and it isn't just another family's
  name (JONES). Most Parts first, skipping terms that are only short codes (`PT`,
  `TT`), only repeat tray-name words, dangle a connective (`FOR HEADED`),
  or overlap a term already taken (`DISC` after `DYNAMIC DISC`).
- Notes for families the bare names confuse:
  - A brand several families share (the first word of their names, when
    it's in capitals: MIS, REFLEX) means, on its own, the one with the most
    Parts -- the brand's main system. That family says so.
  - CATCH_ALL_FAMILIES, named for a kind of part rather than a product
    line, say that parts of a named system aren't theirs.

A description is keyed by the family's display name, the name the
classifiers choose between. Generating them needs every family's Part
descriptions from the graph, so load_family_descriptions caches them in
process; they only change when the catalog is re-seeded (or a display name
is edited, which changes the cache key).
"""
import logging
import re
import time
from collections import Counter

from catalog.repository import ProductFamilyRecord
from retrieval.graph_client import get_graph_client

logger = logging.getLogger(__name__)

# Every classification sends every family's description (18 today), so
# each term is paid for on every turn.
MAX_DISTINCTIVE_TERMS = 6
# Long enough that a turn almost never reads the graph; a re-seed shows up
# within the hour, or at the next deploy.
DESCRIPTIONS_CACHE_SECONDS = 3600

# The one curated input. These families are named for a kind of part
# (screws, instruments, tissue, accessories) rather than a product line, so
# a question about any system's screws or instruments drifts toward them --
# "What screws are in the MIS set?" scored Cannulated Screws above MIS -
# Foot Recon System (issue #30). No catalog signal tells them apart from
# product lines: their names' words are no less common in other families'
# Parts than "Core Foot" or "Mini Foot" are, and "Surgical Accessories"
# shares no word with its Parts at all. Keyed by catalog name, which an
# admin's display-name edit doesn't change.
CATCH_ALL_FAMILIES = frozenset(
    {"Cannulated Screws", "Lab General Instruments", "Soft Tissue", "Surgical Accessories"}
)

# Catalog qualifiers that say which variant of a Part this is, never which
# family: dropped from every segment like sizes are.
_QUALIFIERS = frozenset(
    {
        "ALT", "BX", "CVD", "EA", "LARGE", "LEFT", "LONG", "LRG", "LT", "MEDIUM", "NON-S", "REG",
        "RIGHT", "RT", "SHORT", "SMALL", "STANDARD", "STD", "STERILE", "STR",
    }
)
# Words that join a segment's other words but never make a term on their
# own: a term starting or ending with one is a fragment.
_CONNECTIVES = frozenset({"AND", "FOR", "OF", "OVER", "TO", "USE", "WITH"})
_WORD = re.compile(r"[A-Z][A-Z/-]*")
# A size (`15X15`, `4.5MM`, `T15`, `3/4"`) or a single letter (L, R, T).
_SIZE_OR_LETTER = re.compile(r"[\d\"#()]|^[A-Z]$")


def _words(text: str) -> set[str]:
    # Plurals count as the same word: "Sterile Staple Kits" names STAPLE KIT.
    return {
        word[:-1] if len(word) > 3 and word.endswith("S") else word
        for word in _WORD.findall(text.upper())
    }


def _segment_terms(description: str, family_words: set[str]) -> set[str]:
    """The candidate terms in one Part description, one per segment, with
    the family's own name words left out (REFLEX TETRA -> TETRA)."""
    terms = set()
    for segment in description.upper().split(","):
        words = [
            word
            for word in segment.split()
            if not _SIZE_OR_LETTER.search(word)
            and word not in _QUALIFIERS
            and not _words(word) <= family_words
        ]
        if words:
            terms.add(" ".join(words))
    return terms


def _usable(term: str, tray_words: set[str]) -> bool:
    """Not only short codes (PT, TT), not a fragment (FOR HEADED), and not
    a repeat of the tray names already in the description."""
    words = term.split()
    return (
        any(len(word) >= 3 for word in words)
        and words[0] not in _CONNECTIVES
        and words[-1] not in _CONNECTIVES
        and not _words(term) <= tray_words
    )


def _overlaps(term: str, taken: str) -> bool:
    """One term inside the other (DISC, DYNAMIC DISC), or two variants of
    the same thing (SHANNON RECTA, SHANNON CORTA): the second adds little."""
    words, taken_words = term.split(), taken.split()
    return (
        set(words) <= set(taken_words)
        or set(taken_words) <= set(words)
        or words[0] == taken_words[0]
    )


def _distinctive_terms(
    term_counts: Counter,
    other_families_terms: set[frozenset[str]],
    other_families_names: list[set[str]],
    tray_words: set[str],
) -> list[str]:
    def is_elsewhere(term: str) -> bool:
        # Another family's Parts carry its words in one segment (LOCATOR,
        # inside REFLEX's STAPLE LOCATOR), or it only repeats another
        # family's name (JONES, on a Cannulated Screws depth gauge).
        words = _words(term)
        return any(words <= other_term for other_term in other_families_terms) or any(
            words <= other_name for other_name in other_families_names
        )

    chosen: list[str] = []
    # Most Parts first; on a tie the longer, more specific term.
    ranked = sorted(term_counts, key=lambda term: (-term_counts[term], -len(term.split()), term))
    for term in ranked:
        if is_elsewhere(term) or not _usable(term, tray_words):
            continue
        if any(_overlaps(term, taken) for taken in chosen):
            continue
        chosen.append(term)
        if len(chosen) == MAX_DISTINCTIVE_TERMS:
            break
    return chosen


def _brand(family: ProductFamilyRecord) -> str | None:
    """The first word of the family's name, when it's written in capitals
    like a brand (MIS, REFLEX, IMX) rather than a plain word (Ankle)."""
    name_words = family.display_name.split()
    first_word = name_words[0] if name_words else ""
    return first_word if len(first_word) > 1 and first_word.isalpha() and first_word.isupper() else None


def describe_families(
    families: list[ProductFamilyRecord], part_descriptions_by_family: dict[str, list[str]]
) -> dict[str, str]:
    """Family display name -> its description. `part_descriptions_by_family`
    is keyed by catalog name; a family missing from it (an unreachable
    graph, or one seeded without Parts) is described without Part terms.
    """
    name_words = {family.name: _words(family.display_name) for family in families}
    part_descriptions = {
        family.name: part_descriptions_by_family.get(family.name, []) for family in families
    }
    # Counted once per Part, so a term's count is how many Parts carry it.
    term_counts: dict[str, Counter] = {}
    # Each family's terms with nothing left out: TETRA is distinctive to
    # REFLEX only if no other family has a TETRA segment, whatever its
    # name.
    all_terms: dict[str, set[str]] = {}
    for family in families:
        term_counts[family.name] = Counter()
        all_terms[family.name] = set()
        for description in part_descriptions[family.name]:
            terms = _segment_terms(description, name_words[family.name])
            term_counts[family.name].update(terms)
            all_terms[family.name] |= terms | _segment_terms(description, set())

    families_by_brand: dict[str, list[ProductFamilyRecord]] = {}
    for family in families:
        brand = _brand(family)
        if brand:
            families_by_brand.setdefault(brand, []).append(family)
    brand_main_families = {
        max(branded, key=lambda family: len(part_descriptions[family.name])).name
        for branded in families_by_brand.values()
        if len(branded) > 1 and any(part_descriptions[family.name] for family in branded)
    }
    brands = sorted(families_by_brand)

    descriptions: dict[str, str] = {}
    for family in families:
        others = [other for other in families if other.name != family.name]
        other_families_terms = {
            frozenset(_words(term)) for other in others for term in all_terms[other.name]
        }
        other_families_names = [name_words[other.name] for other in others]

        # A tray named just like its family (MIS Foot Recon System) adds
        # nothing; most families have exactly one such tray.
        trays = [
            tray.display_name
            for tray in family.trays
            if _words(tray.display_name) - name_words[family.name] - {"SYSTEM", "TRAY"}
        ]
        tray_words = {word for tray in family.trays for word in _words(tray.display_name)}
        terms = _distinctive_terms(
            term_counts[family.name], other_families_terms, other_families_names, tray_words
        )

        sentences = []
        if trays:
            sentences.append(f"Trays: {'; '.join(trays)}.")
        if terms:
            sentences.append(f"Parts include {', '.join(term.lower() for term in terms)}.")
        if family.name in brand_main_families:
            sentences.append(f"{_brand(family)} on its own means this family.")
        if family.name in CATCH_ALL_FAMILIES and brands:
            sentences.append(f"Not for {', '.join(brands)} parts.")
        descriptions[family.display_name] = " ".join(sentences)
    return descriptions


_cached_descriptions: dict[str, str] | None = None
_cached_key: tuple | None = None
_cached_at = 0.0


def _cache_key(families: list[ProductFamilyRecord]) -> tuple:
    return tuple(
        (family.name, family.display_name, tuple(tray.display_name for tray in family.trays))
        for family in families
    )


async def load_family_descriptions(families: list[ProductFamilyRecord]) -> dict[str, str]:
    """describe_families over the graph's Parts, cached in process. A graph
    failure is logged and answered with tray-only descriptions, uncached, so
    the next turn tries again: a classifier with thinner descriptions still
    beats a turn that fails.
    """
    global _cached_descriptions, _cached_key, _cached_at
    key = _cache_key(families)
    if (
        _cached_descriptions is not None
        and _cached_key == key
        and time.monotonic() - _cached_at < DESCRIPTIONS_CACHE_SECONDS
    ):
        return _cached_descriptions
    try:
        part_descriptions = await get_graph_client().list_part_descriptions_by_family(
            [family.name for family in families]
        )
    except Exception as error:  # noqa: BLE001 -- any graph failure degrades the same way
        logger.warning("family_descriptions: graph unavailable, describing trays only: %s", error)
        return describe_families(families, {})
    _cached_descriptions = describe_families(families, part_descriptions)
    _cached_key = key
    _cached_at = time.monotonic()
    return _cached_descriptions


def clear_family_descriptions_cache() -> None:
    global _cached_descriptions, _cached_key
    _cached_descriptions = None
    _cached_key = None

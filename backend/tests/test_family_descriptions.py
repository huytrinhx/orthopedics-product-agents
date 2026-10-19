"""Issue #30: each Product Family's description, generated from a small
fixture catalog shaped like the real one (catalog/family_descriptions.py),
and its in-process cache over a faked graph client.
"""
import uuid
from dataclasses import replace

import pytest

from catalog import family_descriptions
from catalog.family_descriptions import (
    describe_families,
    load_family_descriptions,
    load_generated_family_descriptions,
)
from catalog.repository import ProductFamilyRecord, TrayRecord


def _family(name: str, trays: list[str], display_name: str | None = None) -> ProductFamilyRecord:
    return ProductFamilyRecord(
        uuid.uuid4(),
        name,
        display_name or name,
        [TrayRecord(uuid.uuid4(), tray, tray) for tray in trays],
    )


_MIS_FOOT_RECON = _family("MIS - Foot Recon System", ["MIS Foot Recon System"])
_MIS_POWER = _family("MIS - Power System", ["MIS Burrs (Sterile-Packed)", "MIS Power"])
_REFLEX = _family(
    "REFLEX Foot Recon Systems",
    ["REFLEX® Foot Recon Nitinol Implant System", "REFLEX® Sterile Staple Kits"],
)
_HYBRID = _family("REFLEX HYBRID", ["REFLEX® HYBRID Implant System"])
_CANNULATED = _family("Cannulated Screws", ["Forefoot Fixation System", "Ø5.5mm Screw System"])
_JONES = _family("Jones Fracture Screw System", ["Jones Fracture Screw System"])
_FAMILIES = [_MIS_FOOT_RECON, _MIS_POWER, _REFLEX, _HYBRID, _CANNULATED, _JONES]

_PARTS = {
    "MIS - Foot Recon System": [
        "MIS HV CHAMFER FT, 3.0X22MM",
        "MIS HV CHAMFER FT, 3.0X24MM",
        "MIS COMPRESSION PT, 3.5X30MM",
        "GUIDEPIN, MIS 1.4FT, 1.4X150MM",
        "DRIVER, CANNULATED, AO/QC, T10MIS FT",
        "SCREW, CANN, HEADED, 3.0X20MM",
    ],
    "MIS - Power System": [
        "BURR, SHANNON CORTA, 2.0X12MM",
        "BURR, SHANNON RECTA LARGA, 3.1X20MM",
        "ELECTRONIC CORDED MOTOR, MIS POWER, 3M",
    ],
    "REFLEX Foot Recon Systems": [
        "STAPLE, NON-S, REFLEX ULTRA, 15X15",
        "STAPLE, NON-S, REFLEX ULTRA, 18X15",
        "STAPLE, REFLEX TETRA, MTP FUSION, 26MM",
        "STAPLE, REFLEX TETRA, LAPIDUS, 30MM",
        "DYNAMIC DISC, NITINOL, REFLEX, 7MM",
        "COUNTERSINK, CANN, AO/QC, 4.5 DISC",
        "STAPLE KIT, REFLEX ULTRA, 15X15",
        "BLADE, STEP, AKIN",
        "PIN, STAPLE LOCATOR, 1.6MM",
    ],
    "REFLEX HYBRID": [
        "IMPLANT, REFLEX HYBRID, MTP FUSION, L",
        "TEMPLATE, MTP, REFLEX HYBRID, LEFT",
        "PIN, REFLEX LOCATOR, 1.6MM",
    ],
    "Cannulated Screws": [
        "SCREW, CANN, HEADED, 4.0X30MM",
        "SCREW, CANN, HEADLESS, 4.0X30MM",
        "IMPLANT, DIGITAL FUSION, HD, 2.5X20MM",
        "DEPTH GAUGE, JONES, USE WITH 1.6 WIRE",
        "DRILL BIT FOR 5.5MM SCREWS",
    ],
    "Jones Fracture Screw System": ["SCREW, JONES FX, SOLID, 5.5X40MM"],
}


def _described() -> dict[str, str]:
    return describe_families(_FAMILIES, _PARTS)


def test_distinctive_part_terms_drop_sizes_qualifiers_and_the_familys_own_name():
    description = _described()["REFLEX Foot Recon Systems"]
    # Most Parts first (ULTRA 3, TETRA 2), then the longer term on a tie,
    # capped at MAX_DISTINCTIVE_TERMS. REFLEX itself, the sizes, NON-S,
    # STAPLE and NITINOL (the trays already say them), MTP FUSION (HYBRID
    # has it) and DISC (inside DYNAMIC DISC) never make it.
    assert "Parts include ultra, tetra, dynamic disc, staple locator, akin, blade." in description


def test_a_term_another_familys_parts_or_name_carry_is_not_distinctive():
    descriptions = _described()
    # CANN, HEADED and SCREW are on an MIS Part too; JONES is the Jones
    # family's own name, on a Cannulated Screws depth gauge.
    assert "Parts include depth gauge, digital fusion, headless." in descriptions["Cannulated Screws"]
    assert "Parts include hv chamfer ft, compression pt, driver, guidepin." in descriptions[
        "MIS - Foot Recon System"
    ]


def test_a_term_inside_another_familys_segment_is_not_distinctive():
    # HYBRID's LOCATOR (REFLEX stripped) is one word of REFLEX's STAPLE
    # LOCATOR: a "locator pin in the reflex tray" mustn't point at HYBRID.
    assert _described()["REFLEX HYBRID"] == "Trays: REFLEX® HYBRID Implant System. Parts include template."


def test_short_codes_and_dangling_connectives_are_never_terms():
    descriptions = _described()
    # JONES FX less the family's own JONES is FX; DRILL BIT FOR SCREWS less
    # SCREW ends in FOR.
    assert descriptions["Jones Fracture Screw System"] == "Parts include solid."
    assert "drill bit" not in descriptions["Cannulated Screws"]


def test_trays_named_like_their_family_are_left_out():
    descriptions = _described()
    assert "Trays:" not in descriptions["MIS - Foot Recon System"]
    assert descriptions["MIS - Power System"].startswith("Trays: MIS Burrs (Sterile-Packed). ")
    assert descriptions["REFLEX Foot Recon Systems"].startswith(
        "Trays: REFLEX® Foot Recon Nitinol Implant System; REFLEX® Sterile Staple Kits. "
    )


def test_a_shared_brand_on_its_own_means_the_family_with_the_most_parts():
    descriptions = _described()
    assert descriptions["MIS - Foot Recon System"].endswith("MIS on its own means this family.")
    assert descriptions["REFLEX Foot Recon Systems"].endswith("REFLEX on its own means this family.")
    assert "on its own" not in descriptions["MIS - Power System"]
    assert "on its own" not in descriptions["REFLEX HYBRID"]


def test_only_catch_all_families_say_named_systems_parts_arent_theirs():
    descriptions = _described()
    assert descriptions["Cannulated Screws"].endswith("Not for MIS, REFLEX parts.")
    assert all(
        "Not for" not in description
        for name, description in descriptions.items()
        if name != "Cannulated Screws"
    )


def test_descriptions_are_keyed_by_display_name_and_survive_missing_parts():
    renamed = _family("REFLEX HYBRID", ["REFLEX® HYBRID Implant System"], display_name="Reflex Hybrid")

    descriptions = describe_families([renamed, _JONES], {})

    assert descriptions == {
        "Reflex Hybrid": "Trays: REFLEX® HYBRID Implant System.",
        "Jones Fracture Screw System": "",
    }
    assert describe_families([_MIS_POWER], {})["MIS - Power System"] == "Trays: MIS Burrs (Sterile-Packed)."


class _FakeGraphClient:
    def __init__(self, parts: dict[str, list[str]] | Exception):
        self.parts = parts
        self.calls = 0

    async def list_part_descriptions_by_family(self, product_families: list[str]) -> dict[str, list[str]]:
        self.calls += 1
        if isinstance(self.parts, Exception):
            raise self.parts
        return {name: self.parts[name] for name in product_families if name in self.parts}


@pytest.fixture
def graph(monkeypatch):
    family_descriptions.clear_family_descriptions_cache()
    client = _FakeGraphClient(_PARTS)
    monkeypatch.setattr(family_descriptions, "get_graph_client", lambda: client)
    yield client
    family_descriptions.clear_family_descriptions_cache()


async def test_descriptions_are_cached_until_the_families_change(graph):
    first = await load_family_descriptions(_FAMILIES)
    assert await load_family_descriptions(_FAMILIES) == first
    assert graph.calls == 1

    # A display-name edit (or a re-seed's trays) is a new key.
    renamed = [*_FAMILIES[:-1], _family("Jones Fracture Screw System", [], display_name="Jones Screws")]
    assert "Jones Screws" in await load_family_descriptions(renamed)
    assert graph.calls == 2


async def test_the_cache_expires(graph, monkeypatch):
    await load_family_descriptions(_FAMILIES)
    monkeypatch.setattr(family_descriptions, "DESCRIPTIONS_CACHE_SECONDS", 0)
    await load_family_descriptions(_FAMILIES)
    assert graph.calls == 2


async def test_an_unreachable_graph_falls_back_to_trays_and_isnt_cached(graph):
    graph.parts = OSError("connection refused")

    descriptions = await load_family_descriptions(_FAMILIES)

    assert descriptions["MIS - Power System"] == "Trays: MIS Burrs (Sterile-Packed)."
    graph.parts = _PARTS
    assert await load_family_descriptions(_FAMILIES) == _described()
    assert graph.calls == 2


async def test_an_admins_description_replaces_the_generated_one_on_the_next_turn(graph):
    generated = await load_family_descriptions(_FAMILIES)
    edited = replace(
        _REFLEX, description_override="Nitinol staples: TETRA, ULTRA, MINI; Dynamic Disc; stepped blades."
    )
    families = [edited if family.name == edited.name else family for family in _FAMILIES]

    described = await load_family_descriptions(families)

    assert described["REFLEX Foot Recon Systems"] == edited.description_override
    assert {
        name: description for name, description in described.items() if name != "REFLEX Foot Recon Systems"
    } == {name: description for name, description in generated.items() if name != "REFLEX Foot Recon Systems"}
    # The cached generated text stays what Settings shows beside the edit,
    # and the edit didn't wait for the cache.
    assert (await load_generated_family_descriptions(families)) == generated
    assert graph.calls == 1

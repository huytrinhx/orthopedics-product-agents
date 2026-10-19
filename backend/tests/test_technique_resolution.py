"""Issue #28's step item -> SKU set resolution (techniques/resolution.py),
on a fixture catalog of real master-catalog rows. The cases are the
hand-built expectations from the 2026-10-18 evaluation: MIS's proximal
screw step, Lisfranc's step 5 by screw diameter, and REFLEX's TETRA +
ULTRA Lapidus."""
from ingestion.technique_extraction import NamedValue, StepItemReference
from techniques.resolution import (
    CATALOG_ONLY,
    IN_DOCUMENT,
    LITERAL,
    UNRESOLVED,
    CatalogPart,
    ResolutionContext,
    applies_to,
    normalize_parameter_value,
    resolve_item,
    resolve_step_items,
    resolve_tray_references,
    scope_families,
)

MIS = "MIS - Foot Recon System"
ANCILLARY = "Ancillary Foot Recon System"
REFLEX = "REFLEX Foot Recon Systems"
CORE_FOOT = "Core Foot Plating System"
ANKLE_FUSION = "Ankle Fusion Plating System"
CANNULATED = "Cannulated Screws"


def _screw(sku, description, family, thread, guidewire, pre_drill, driver):
    return CatalogPart(
        sku, description, frozenset({family}), "Implant", thread, guidewire, pre_drill, driver
    )


def _part(sku, description, *families):
    return CatalogPart(sku, description, frozenset(families), "Disposable")


_PARTS = [
    *[
        _screw(f"MSCF40{length}", f"MIS HV CHAMFER FT, 4.0 X {length}", MIS, "Full",
               "Ø1.6 x 200mm (Cobalt Chrome)", "Ø3.6mm", "T15")
        for length in range(26, 67, 2)
    ],
    _screw("MSCF3520", "MIS HV CHAMFER FT, 3.5 X 20", MIS, "Full", "Ø1.4 x 200mm", "Ø2.9mm", "T15"),
    _screw("MSCP3020", "MIS COMPRESSION PT, 3.0 X 20", MIS, "Partial", "Ø1.2 x 150mm", "Ø2.3mm", "T10"),
    _part("MGW16200", "GUIDEPIN, MIS 4.0 FT, 1.6 X 200MM", MIS),
    _part("MGW14200", "GUIDEPIN, MIS 3.5 FT, 1.4 X 200MM", MIS),
    _part("MSN40029", "DEPTH GAUGE, MIS, USE WITH 200MM WIRE", MIS),
    _part("MSN40027", "TISSUE PROTECTOR, MIS, 3.5MM", MIS),
    _part("MSN40028", "TISSUE PROTECTOR, MIS, 4.0MM", MIS),
    _part("MSN10020", "DRILL BIT, CANN, AO/QC, 2.9 MIS", MIS),
    _part("MSN10021", "DRILL BIT, CANN, AO/QC, 3.6 MIS", MIS),
    _part("MSN20017", "COUNTERSINK, CANN, AO/QC, 3.5 MIS FT", MIS),
    _part("MSN20018", "COUNTERSINK, CANN, AO/QC, 4.0 MIS FT", MIS),
    _part("MSN30008", "DRIVER, CANNULATED, AO/QC, T10 MIS FT", MIS),
    _part("MSN30009", "DRIVER, CANNULATED, AO/QC, T15 MIS FT", MIS),
    # Another family's 3.6mm drill and T15 driver: in scope only when the
    # document's index finds that family.
    _part("MSN10013", "DRILL BIT, CANN, AO/QC, 3.6MM, 5.5 SCRW", CANNULATED),
    _part("MSN30005", "DRIVER, CANNULATED, AO/QC, T15", CANNULATED),
    _screw("MSDL3720", "SCREW, LISFRANC, SOLID, 3.7 X 20MM", ANCILLARY, "Full", "Ø1.4x150mm", "Ø2.6mm", "T15"),
    _screw("MSDL4120", "SCREW, LISFRANC, SOLID, 4.1 X 20MM", ANCILLARY, "Full", "Ø1.4x150mm", "Ø2.8mm", "T15"),
    _part("MSN10015", "DRILL BIT, CANN, AO/QC, 2.6MM, 3.7 PILOT", ANCILLARY),
    _part("MSN10016", "DRILL BIT, CANN, AO/QC, 3.7MM, 3.7 OVER", ANCILLARY),
    _part("MSN10017", "DRILL BIT, CANN, AO/QC, 2.8MM, 4.1 PILOT", ANCILLARY),
    _part("MSN10018", "DRILL BIT, CANN, AO/QC, 4.1MM, 4.1 OVER", ANCILLARY),
    _part("MSN20014", "COUNTERSINK, CANN, LISFRANC", ANCILLARY),
    _part("MSTG3000", "DRILL GUIDE, STAPLE, TETRA, 30MM WIDE", REFLEX),
    _part("MSTG3023", "TARGETING GUIDE, 30 DORSAL X 23 MEDIAL", REFLEX),
    _part("MSG14150", "GUIDEWIRE, NON-THREADED, 1.4 x 150MM", REFLEX, ANCILLARY),
    _part("MSTD1030", "DRILL BIT, STAPLE, AO/QC, SHORT, 3.0MM", REFLEX),
    _part("MSTD1130", "DRILL BIT, STAPLE, AO/QC, LONG, 3.0MM", REFLEX),
    _part("MSTP0030", "PIN, STAPLE LOCATOR, 3.0MM", REFLEX),
    _part("MSTN0003", "SLEEVE, STAPLE TARGETING, 90 DEG", REFLEX),
    _part("MSTN0002", "INSERTER, STAPLE, REFLEX MAX/ULTRA", REFLEX),
    _part("MSTX3002NS", "STAPLE, REFLEX TETRA, LAPIDUS, 30MM", REFLEX),
    _part("MST52316NS", "STAPLE, NON-S, REFLEX ULTRA, 23 X 16MM", REFLEX),
    _part("MSTTAMP1", "TAMP, STAPLE", REFLEX),
    _part("MSN90006", "HANDLE, AO/QC, NON-RATCHET, CANN, STD", REFLEX, ANKLE_FUSION),
    _part("MPN52012", "ARTHROTOME, ANKLE JOINT, 6.5MM, AO/QC", ANKLE_FUSION),
    _part("MSTC0000", "CURETTE, CUP, STRAIGHT, AO/QC", REFLEX, CORE_FOOT),
    _part("MPDP0020", "DRILL PIN, JOINT FENESTRATION, 2.0MM", REFLEX, CORE_FOOT),
    _part("MPP2100U", "PLATE, LAPIDUS STANDARD", CORE_FOOT),
]
CATALOG = {part.sku: part for part in _PARTS}


def _item(mention, item_type, specs=None, applies_when=None, sku=None):
    return StepItemReference(
        mention=mention,
        item_type=item_type,
        sku=sku,
        specs=[NamedValue(name=name, value=value) for name, value in (specs or {}).items()],
        applies_when=[
            NamedValue(name=name, value=value) for name, value in (applies_when or {}).items()
        ],
    )


def _context(family, indexed_skus, required_tray_families=()):
    sku_chunks = {sku: [14] for sku in indexed_skus}
    return ResolutionContext(
        catalog=CATALOG,
        scope=scope_families(family, sku_chunks, CATALOG, required_tray_families),
        document_sku_chunks=sku_chunks,
    )


def _family_skus(family):
    return [part.sku for part in _PARTS if family in part.families]


def test_mis_proximal_screw_step_resolves_to_the_golden_set():
    context = _context(MIS, _family_skus(MIS))
    items = [
        _item("Ø4.0mm FT screw", "screw", {"diameter": "4.0mm", "thread": "FT"}),
        _item("Ø1.6mm guidepin", "guide pin", {"diameter": "Ø1.6mm"}),
        _item("depth gauge", "depth gauge"),
        _item("Ø4.0mm tissue protector", "tissue protector", {"diameter": "4.0mm"}),
        _item("drill", "drill bit", {"for_screw_diameter": "4.0mm", "for_screw_thread": "FT"}),
        _item("countersink", "countersink", {"for_screw_diameter": "4.0mm"}),
        # No spec at all: resolves through the step's own screws' driver spec.
        _item("appropriate driver", "driver"),
    ]

    resolved = resolve_step_items(items, context, [8])

    assert {sku for item in resolved for sku in item.skus} == {
        "MGW16200", "MSN40029", "MSN40028", "MSN10021", "MSN20018", "MSN30009",
        *(f"MSCF40{length}" for length in range(26, 67, 2)),
    }
    assert {item.resolution for item in resolved} == {IN_DOCUMENT}
    # The step's chunk first, then the table chunk the index found it in.
    assert resolved[0].source_chunk_indexes == (8, 14)


def test_lisfranc_step_5_is_a_parameterised_set_by_screw_diameter():
    context = _context(ANCILLARY, _family_skus(ANCILLARY))
    items = [
        _item("Ø2.6mm for Ø3.7mm screws", "drill bit", {"for_screw_diameter": "3.7mm"},
              {"screw_diameter": "3.7mm"}),
        _item("Ø2.8mm for Ø4.1mm screws", "drill bit", {"for_screw_diameter": "4.1mm"},
              {"screw_diameter": "4.1mm"}),
        _item("Ø3.7mm over-drill", "drill bit", {"diameter": "3.7mm"}, {"screw_diameter": "3.7mm"}),
        _item("Ø4.1mm over-drill", "drill bit", {"diameter": "4.1mm"}, {"screw_diameter": "4.1mm"}),
        _item("countersink", "countersink"),
    ]

    resolved = resolve_step_items(items, context, [20])
    by_option = {
        option: sorted(
            sku
            for item, result in zip(items, resolved)
            if applies_to(item.parameter_values(), {"screw_diameter": option})
            for sku in result.skus
        )
        for option in ("3.7", "4.1")
    }

    # "Ø2.6mm" belongs to MSN10015 ("2.6MM, 3.7 PILOT"), never the 3.7mm
    # over-drill: a part's diameter is its own leading dimension.
    assert by_option == {
        "3.7": ["MSN10015", "MSN10016", "MSN20014"],
        "4.1": ["MSN10017", "MSN10018", "MSN20014"],
    }


def test_reflex_lapidus_resolves_cross_family_parts_the_index_found():
    indexed = _family_skus(REFLEX) + ["MPN52012"]
    context = _context(REFLEX, indexed)
    items = [
        _item("30mm TETRA drill guide", "drill guide", {"width": "30mm"}),
        _item("30 x 23 medial targeting guide", "targeting guide", {"width": "30 x 23"}),
        _item("Ø1.4 x 150mm wires", "guidewire", {"diameter": "1.4mm", "length": "150mm"}),
        _item("short Ø3.0mm drill", "drill bit", {"diameter": "3.0mm", "length": "short"}),
        _item("long Ø3.0mm drill", "drill bit", {"diameter": "3.0mm", "length": "long"}),
        _item("locator pins", "locator pin"),
        _item("drill sleeve", "sleeve"),
        _item("staple inserter", "inserter"),
        _item("30mm TETRA staple", "staple", {"width": "30mm"}),
        _item("23 x 16 ULTRA staple", "staple", {"width": "23 x 16mm"}),
        _item("tamp", "tamp"),
        _item("arthrotome", "arthrotome"),
        _item("curette", "curette"),
        _item("fenestration drill pins", "drill pin"),
        _item("fixed AO handle", "handle"),
    ]

    resolved = resolve_step_items(items, context, [30])

    assert [list(item.skus) for item in resolved] == [
        ["MSTG3000"], ["MSTG3023"], ["MSG14150"], ["MSTD1030"], ["MSTD1130"], ["MSTP0030"],
        ["MSTN0003"], ["MSTN0002"], ["MSTX3002NS"], ["MST52316NS"], ["MSTTAMP1"],
        ["MPN52012"], ["MSTC0000"], ["MPDP0020"], ["MSN90006"],
    ]


def test_scope_is_the_family_plus_indexed_families_plus_required_trays():
    assert scope_families(REFLEX, ["MPN52012", "NOT-A-SKU"], CATALOG, [CORE_FOOT]) == {
        REFLEX, ANKLE_FUSION, CORE_FOOT
    }
    # Out of scope, the other family's plate isn't a candidate at all ...
    assert resolve_item(_item("Lapidus plate", "plate"), _context(REFLEX, [])).resolution == UNRESOLVED
    # ... and a required tray's family brings it in, as a catalog-only
    # candidate the document's own index never found.
    widened = resolve_item(_item("Lapidus plate", "plate"), _context(REFLEX, [], [CORE_FOOT]))
    assert (widened.skus, widened.resolution) == (("MPP2100U",), CATALOG_ONLY)


def test_the_document_index_breaks_ties_with_other_families_parts():
    # Both 3.6mm drills are in scope (the index found a Cannulated Screws
    # part), but only the MIS one is in the document.
    context = _context(MIS, ["MSCF4026", "MSN10021", "MSN30005"])

    drill = resolve_item(_item("Ø3.6mm drill", "drill bit", {"diameter": "3.6mm"}), context)

    assert (drill.skus, drill.resolution) == (("MSN10021",), IN_DOCUMENT)


def test_a_literal_sku_is_taken_as_printed_and_an_unknown_one_resolves_by_specs():
    context = _context(MIS, [])
    literal = resolve_item(_item("MSN30009 driver", "driver", sku="MSN30009"), context, step_chunk_indexes=[3])
    assert (literal.skus, literal.resolution, literal.source_chunk_indexes) == (("MSN30009",), LITERAL, (3,))

    guessed = resolve_item(_item("T10 driver", "driver", {"driver": "T10"}, sku="MSN99999"), context)
    assert (guessed.skus, guessed.resolution) == (("MSN30008",), CATALOG_ONLY)


def test_a_spec_nothing_matches_leaves_the_item_unresolved():
    resolved = resolve_item(_item("Ø9.9mm drill", "drill bit", {"diameter": "9.9mm"}), _context(MIS, []))
    assert (resolved.skus, resolved.resolution) == ((), UNRESOLVED)


def test_tray_references_match_catalog_names_ignoring_case_and_trademarks():
    trays_by_family = {
        CORE_FOOT: ["Core Foot Recon Plating System"],
        REFLEX: ["REFLEX® Foot Recon Nitinol Implant System", "REFLEX® MINI Nitinol Staple System"],
    }

    assert resolve_tray_references(
        ["core foot recon plating system", "REFLEX Foot Recon Nitinol Implant System", "Unknown Tray"],
        trays_by_family,
    ) == ["Core Foot Recon Plating System", "REFLEX® Foot Recon Nitinol Implant System"]
    # A family's name means all of its trays.
    assert resolve_tray_references(["REFLEX Foot Recon Systems"], trays_by_family) == trays_by_family[REFLEX]


def test_parameter_values_compare_by_number_and_unbound_parameters_keep_every_option():
    assert normalize_parameter_value("Ø3.7mm") == normalize_parameter_value("3.70") == "3.7"
    assert normalize_parameter_value("FT") == "ft"
    assert applies_to({"screw_diameter": "3.7mm"}, {"Screw Diameter": "3.7"})
    assert not applies_to({"screw_diameter": "3.7mm"}, {"screw_diameter": "4.1"})
    assert applies_to({"screw_diameter": "3.7mm"}, {"bridge_size": "30mm"})
    assert applies_to({}, {"screw_diameter": "4.1"})

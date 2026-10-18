"""Exercises backend/agents/fact_check.py -- pure functions, no LLM, no DB."""
from agents.fact_check import check_answer, format_issues_for_correction

_DOCUMENT_ID = "0b6f4a0e-3c1d-4e8a-9f2b-5d7c8e9a1b2c"
_FULL_THREAD = {"sku": "MSK42020", "description": "MIS HV CHAMFER FT 4.0 X 20", "thread": "Full"}
_PARTIAL_THREAD = {"sku": "MSK32016", "description": "MIS COMPRESSION PT 3.0 X 16", "thread": "Partial (1/3)"}
_SMOOTH = {"sku": "MGP14150", "description": "GUIDEPIN 1.4 X 150", "thread": "Smooth"}


def _check(answer: str, *, citation_ids=frozenset(), source_text: str = "", parts=()) -> list[str]:
    return check_answer(answer, citation_ids=set(citation_ids), source_text=source_text, parts=list(parts))


def test_a_grounded_answer_passes():
    answer = (
        f"- MSK42020: 4.0 x 20mm Full Thread screw [{_DOCUMENT_ID}#3]\n"
        "- MSK32016: 3.0 x 16mm PT screw"
    )
    assert _check(answer, citation_ids={f"{_DOCUMENT_ID}#3"}, parts=[_FULL_THREAD, _PARTIAL_THREAD]) == []


def test_flags_a_citation_to_a_passage_that_was_never_retrieved():
    issues = _check(f"Torque to 1.2 Nm [{_DOCUMENT_ID}#9].", citation_ids={f"{_DOCUMENT_ID}#3"})
    assert issues == [f"Cited [{_DOCUMENT_ID}#9], but no passage with that id was retrieved this turn."]


def test_flags_a_sku_that_appears_nowhere_in_the_sources():
    issues = _check("Use MSK99999 for this.", source_text="MSK42020 is in the tray", parts=[_FULL_THREAD])
    assert issues == ["MSK99999 doesn't appear in any catalog record or passage retrieved this turn."]


def test_a_sku_found_only_in_passage_text_passes():
    assert _check("Use MPPA100L for this.", source_text="PLATE MPPA100L, LATERAL FIBULA") == []


def test_ignores_tokens_that_only_look_like_skus():
    # Too short, no digit, or a measurement -- none are SKU-shaped.
    assert _check("The MIS FT screw is 4.0mm, uses a T8 driver, part of REFLEX.") == []


def test_ignores_hex_inside_a_citation_uuid():
    answer = "Torque to 1.2 Nm [ABCDEF12-3c1d-4e8a-9f2b-5d7c8e9a1b2c#0]."
    assert _check(answer, citation_ids={"ABCDEF12-3c1d-4e8a-9f2b-5d7c8e9a1b2c#0"}) == []


def test_flags_a_thread_type_the_catalog_contradicts():
    issues = _check("- MSK42020: 4.0 x 20mm PT screw", parts=[_FULL_THREAD])
    assert issues == [
        (
            "MSK42020 is Full Thread (FT) in the catalog (thread: Full), but the answer calls it "
            "Partial Thread (PT)."
        )
    ]


def test_flags_a_spelled_out_thread_type_too():
    issues = _check("MSK32016 is a fully threaded... no, a full-thread screw.", parts=[_PARTIAL_THREAD])
    assert len(issues) == 1
    assert "MSK32016 is Partial Thread (PT)" in issues[0]


def test_skips_a_line_naming_parts_with_different_thread_types():
    # Can't tell which claim belongs to which SKU, so it doesn't guess.
    assert _check("MSK42020 (PT) pairs with MSK32016 (FT)", parts=[_FULL_THREAD, _PARTIAL_THREAD]) == []


def test_skips_parts_without_a_full_or_partial_thread():
    assert _check("MGP14150 is a PT guidepin", parts=[_SMOOTH]) == []


def test_only_checks_the_line_the_sku_is_on():
    answer = "- MSK42020: 4.0 x 20mm\n- The Akin uses a PT screw"
    assert _check(answer, parts=[_FULL_THREAD]) == []


def test_a_conflicting_sources_line_in_the_prompted_format_passes():
    # generate is told to call out a passage that contradicts the catalog
    # with both values on one line -- that line names the passage's wrong
    # thread type, and must not be "corrected" away by the thread check.
    answer = (
        "- MSK42020: 4.0 x 20mm Full Thread screw\n\n"
        "Conflicting sources\n"
        f"MSK42020: the catalog lists Full Thread; [{_DOCUMENT_ID}#3] says partial thread. "
        "This answer uses the catalog value."
    )
    assert _check(answer, citation_ids={f"{_DOCUMENT_ID}#3"}, parts=[_FULL_THREAD]) == []


def test_a_conflict_line_missing_the_catalog_value_is_flagged():
    # Why the prompt asks for both values on the same line: the passage's
    # value on its own reads as the answer's claim.
    issues = _check(f"[{_DOCUMENT_ID}#3] says MSK42020 is partial thread.", citation_ids={f"{_DOCUMENT_ID}#3"}, parts=[_FULL_THREAD])
    assert len(issues) == 1
    assert "MSK42020 is Full Thread (FT)" in issues[0]


# Issue #21: trays as part_lookup returns them -- display names, one or more.
_MINI_TRAY = "REFLEX® MINI Implant System"
_NITINOL_TRAY = "REFLEX® Nitinol Implant System"
_AKIN_BLADE = {"sku": "RFX25001", "description": "BLADE, AKIN, STEPPED", "trays": [_MINI_TRAY]}
_TETRA_STAPLE = {"sku": "RFX18020", "description": "STAPLE, TETRA, 18 X 20", "trays": [_NITINOL_TRAY]}
_PICKUPS = {"sku": "MSN90003", "description": "PICKUPS", "trays": [_MINI_TRAY, _NITINOL_TRAY]}


def test_a_tray_matching_the_catalog_passes_even_without_its_trademark():
    assert _check("RFX25001 is in the REFLEX MINI Implant System tray.", parts=[_AKIN_BLADE, _TETRA_STAPLE]) == []


def test_flags_a_tray_the_catalog_contradicts():
    issues = _check(f"RFX25001 is in the {_NITINOL_TRAY} tray.", parts=[_AKIN_BLADE, _TETRA_STAPLE])
    assert issues == [
        f"RFX25001 is in {_MINI_TRAY} in the catalog, but the answer places it in {_NITINOL_TRAY}."
    ]


def test_a_part_in_several_trays_passes_naming_any_of_them():
    assert _check(f"MSN90003 ships in the {_NITINOL_TRAY}.", parts=[_PICKUPS]) == []


def test_skips_a_line_naming_parts_from_different_trays():
    # Can't tell which tray belongs to which SKU, so it doesn't guess.
    answer = f"RFX25001 and RFX18020 both come from the {_NITINOL_TRAY}."
    assert _check(answer, parts=[_AKIN_BLADE, _TETRA_STAPLE]) == []


def test_only_checks_the_line_the_sku_is_on_for_trays():
    answer = f"- RFX25001: Akin stepped blade\n- Staples live in the {_NITINOL_TRAY}"
    assert _check(answer, parts=[_AKIN_BLADE, _TETRA_STAPLE]) == []


def test_ignores_a_tray_it_was_never_told_about():
    # Only the turn's own Parts' trays, plus any tray_names given, are
    # recognized as tray names.
    assert _check("RFX25001 is in the Sterile Staple Kits.", parts=[_AKIN_BLADE]) == []


def test_flags_a_sibling_tray_given_as_a_known_tray_name():
    issues = check_answer(
        "RFX25001 is in the REFLEX Sterile Staple Kits.",
        citation_ids=set(),
        source_text="",
        parts=[_AKIN_BLADE],
        tray_names={_MINI_TRAY, "REFLEX® Sterile Staple Kits"},
    )
    assert issues == [
        f"RFX25001 is in {_MINI_TRAY} in the catalog, but the answer places it in REFLEX® Sterile Staple Kits."
    ]


def test_a_conflicting_sources_tray_line_passes():
    answer = (
        f"RFX25001: the catalog lists {_MINI_TRAY}; [{_DOCUMENT_ID}#3] says {_NITINOL_TRAY}. "
        "This answer uses the catalog value."
    )
    assert _check(answer, citation_ids={f"{_DOCUMENT_ID}#3"}, parts=[_AKIN_BLADE, _TETRA_STAPLE]) == []


def test_correction_note_lists_every_issue():
    note = format_issues_for_correction(["first problem", "second problem"])
    assert "- first problem\n- second problem" in note

"""Exercises backend/ingestion/chunking.py -- pure logic, no external
dependencies (no DB, no LLM) needed to test it.
"""
import tiktoken

from ingestion.chunking import chunk_document, strip_page_markers
from ingestion.text_extraction import PAGE_MARKER_TEMPLATE

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _words(word_count: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{index}" for index in range(word_count))


def _tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def test_short_plain_text_becomes_a_single_chunk():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunk_document(text, chunk_size=800, overlap=100)

    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["section_title"] is None
    assert "First paragraph." in chunks[0]["content"]
    assert "Second paragraph." in chunks[0]["content"]


def test_paragraphs_pack_until_the_token_budget_then_split():
    paragraphs = [_words(20, f"p{index}_") for index in range(10)]
    text = "\n\n".join(paragraphs)
    chunk_size = _tokens(paragraphs[0]) * 3  # roughly 3 paragraphs per window

    chunks = chunk_document(text, chunk_size=chunk_size, overlap=5)

    assert len(chunks) > 1
    for paragraph in paragraphs:
        assert any(paragraph in chunk["content"] for chunk in chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_share_overlap_content():
    paragraphs = [_words(20, f"p{index}_") for index in range(6)]
    text = "\n\n".join(paragraphs)
    chunk_size = _tokens(paragraphs[0]) * 2
    overlap = _tokens(paragraphs[0])

    chunks = chunk_document(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) >= 2
    last_paragraph_of_first_chunk = chunks[0]["content"].split("\n\n")[-1]
    assert last_paragraph_of_first_chunk in chunks[1]["content"]


def test_oversized_single_paragraph_is_token_windowed():
    huge = _words(3000)

    chunks = chunk_document(huge, chunk_size=800, overlap=100)

    assert len(chunks) > 1
    for chunk in chunks:
        assert _tokens(chunk["content"]) <= 800
        assert chunk["section_title"] is None


def test_headings_split_into_sections_with_titles():
    text = (
        "# Sizing Guide\n"
        "Use the 4.0mm screw for standard bone density.\n\n"
        "# Sterilization\n"
        "Autoclave at 270F for 4 minutes."
    )

    chunks = chunk_document(text)

    assert [chunk["section_title"] for chunk in chunks] == ["Sizing Guide", "Sterilization"]
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1]
    assert "Use the 4.0mm screw" in chunks[0]["content"]
    assert "Autoclave" in chunks[1]["content"]


def test_blank_or_empty_text_yields_no_chunks():
    assert chunk_document("") == []
    assert chunk_document("   \n\n   ") == []


def _marker(page_number: int) -> str:
    return PAGE_MARKER_TEMPLATE.format(page_number=page_number)


def test_plain_text_with_no_markers_has_no_page_numbers():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunk_document(text)

    assert all(chunk["page_number"] is None for chunk in chunks)


def test_chunk_is_tagged_with_the_page_its_content_starts_on():
    text = (
        f"{_marker(1)}\n\n"
        "# Intro\nPage one content.\n\n"
        f"{_marker(2)}\n\n"
        "# Prep\nPage two content."
    )

    chunks = chunk_document(text)

    assert [chunk["page_number"] for chunk in chunks] == [1, 2]
    assert [chunk["section_title"] for chunk in chunks] == ["Intro", "Prep"]


def test_page_marker_survives_when_a_heading_is_the_pages_first_line():
    # Regression: a page marker sits *before* the heading line that starts
    # that page's real content (the common real-world shape -- a PDF page
    # that opens directly with a section title). The marker must still be
    # attributed to the section the heading starts, not lost to a
    # disconnected pre-heading section.
    text = (
        f"{_marker(5)}\n\n"
        "# Sterilization\nAutoclave at 270F for 4 minutes."
    )

    chunks = chunk_document(text)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 5
    assert chunks[0]["section_title"] == "Sterilization"


def test_multiple_pages_packed_into_one_window_keep_the_starting_page():
    text = (
        "# Overview\n"
        f"{_marker(5)}\n\nShort a.\n\n"
        f"{_marker(6)}\n\nShort b.\n\n"
        f"{_marker(7)}\n\nShort c."
    )

    chunks = chunk_document(text)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 5


def test_strip_page_markers_removes_marker_paragraphs_only():
    text = f"{_marker(1)}\n\n# Intro\nPage one content.\n\n{_marker(2)}\n\nPage two content."

    stripped = strip_page_markers(text)

    assert "ORTHOMATE_PAGE_MARKER" not in stripped
    assert "Page one content." in stripped
    assert "Page two content." in stripped


# Issue #26: inputs below are shaped on production's Surgical Technique
# chunk text (entity-extraction evaluation, 2026-10-18), as text_extraction
# would emit it before chunking.


def _titles(chunks: list[dict]) -> list[str | None]:
    return [chunk["section_title"] for chunk in chunks]


def _paths(chunks: list[dict]) -> list[list[str]]:
    return [chunk["heading_path"] for chunk in chunks]


def _chunk_containing(chunks: list[dict], text: str) -> dict:
    [match] = [chunk for chunk in chunks if text in chunk["content"]]
    return match


def test_figure_numbers_never_become_section_titles():
    # Core Foot c9-c11, c15, c18-c20, c32: "1", "2", "3", "8a 9", "10",
    # "11", "12 12", "69c" and "97" came out as titles, and the real
    # "Step 2" heading above figure "1" was dropped.
    text = (
        f"{_marker(5)}\n\n"
        "# Step 1 | Surgical approach\n"
        "Perform a dorsal longitudinal incision.\n"
        "# Step 2 | Joint preparation\n"
        "# 1\n"
        "Expose the articulating surfaces.\n"
        "Starting with the ø22 mm metatarsal (cup) reamer, place over the wire (Fig. 1).\n"
        "# 2\n"
        "Place the ø16 mm phalangeal (cone) reamer over the wire (Fig. 2).\n"
        "The fenestration drill pin may be used to perforate the reamed surfaces\n"
        "# 3\n"
        "(Fig. 3).\n"
        f"\n{_marker(6)}\n\n"
        "# Step 3 | Plate verification and provisional fixation\n"
        "# 8a 9\n"
        "Thread the appropriate locking drill guide into the distal plate holes.\n"
        "# 10\n"
        "Insert a non-locking screw into the compression slot, removing the\n"
        "# 11\n"
        "temporary fixation pin prior to fully tightening the screw (Fig. 11).\n"
        "# 12 12\n"
        "Drill just past the far cortex for the remaining proximal screws (Fig. 12).\n"
        "# 69c\n"
        "Insert the half-thread ø3.5 mm solid cross-plate screw (Fig. 6c).\n"
        "# 97\n"
        "Finish the construct by placing ø3.5 mm locking screws (Fig. 7)."
    )

    chunks = chunk_document(text)

    assert _titles(chunks) == [
        "Step 1 | Surgical approach",
        "Step 2 | Joint preparation",
        "Step 3 | Plate verification and provisional fixation",
    ]
    joint_preparation = chunks[1]["content"]
    assert "Expose the articulating surfaces." in joint_preparation
    assert "perforate the reamed surfaces\n3\n(Fig. 3)." in joint_preparation
    # The sentence the fake "11" heading cut in two is whole again.
    assert "removing the\n11\ntemporary fixation pin" in chunks[2]["content"]
    assert "8a 9" in chunks[2]["content"]


def test_split_headings_are_rejoined_and_letter_spacing_repaired():
    # MIS: "Preparation & Insertion" (twice), "& Wire Placement" and
    # "& Insertion" lost the step line above them; "R emove" and "1 st"
    # came out letter-spaced.
    text = (
        f"{_marker(4)}\n\n"
        "# Step 2. Perform 1 st Metatarsal Osteotomy\n"
        "Insert a burr through the percutaneous hole.\n"
        f"\n{_marker(7)}\n\n"
        "# Step 6. Proximal Screw\n"
        "# Preparation & Insertion\n"
        "Slide the depth gauge over the proximal wire. Remove proximal wire when screw is placed.\n"
        f"\n{_marker(9)}\n\n"
        "# Step 8. R emove Medial Eminence\n"
        "If necessary, remove distal medial eminence using a burr or rongeur.\n"
        "# Step 9. Akin Osteotomy\n"
        "# & Wire Placement\n"
        "Create an Akin (wedge) osteotomy of the proximal phalanx using the burr.\n"
        "# Step 10. Akin Screw Preparation\n"
        "# & Insertion\n"
        "Slide the depth gauge over the wire and down to the bone."
    )

    chunks = chunk_document(text)

    assert _titles(chunks) == [
        "Step 2. Perform 1st Metatarsal Osteotomy",
        "Step 6. Proximal Screw Preparation & Insertion",
        "Step 8. Remove Medial Eminence",
        "Step 9. Akin Osteotomy & Wire Placement",
        "Step 10. Akin Screw Preparation & Insertion",
    ]


def test_letter_spacing_is_only_repaired_into_words_the_document_uses():
    text = "# Notes\nA burr may be used. Position the C arm. NOTE: P rocedure requires a procedure tray."

    [chunk] = chunk_document(text)

    assert "A burr may be used. Position the C arm." in chunk["content"]
    assert "NOTE: Procedure requires" in chunk["content"]


def test_doubled_font_headings_collapse_and_running_footers_stay_body():
    # Core Foot c5 "enhanced enhanced enhanced enhanced compression. ..." and
    # the "4 MEDLINE UNITE" / "MEDLINE MEDLINE UNITE UNITE 3 3" footers that
    # became titles.
    text = (
        f"{_marker(3)}\n\n"
        "# enhanced enhanced enhanced enhanced compression. compression. compression. compression.\n"
        "LAPIDUS STANDARD MPP2100U\n"
        "# MEDLINE MEDLINE UNITE UNITE 3 3\n"
        f"\n{_marker(4)}\n\n"
        "# Step 1. Surgical Approach\n"
        "Make a dorsomedial incision.\n"
        "MEDLINE UNITE 4\n"
        f"\n{_marker(5)}\n\n"
        "# 4 MEDLINE UNITE\n"
        "# Step 2. Joint Reduction\n"
        "Reduce the joint."
    )

    chunks = chunk_document(text)

    assert _titles(chunks) == ["enhanced compression.", "Step 1. Surgical Approach", "Step 2. Joint Reduction"]
    assert "MEDLINE MEDLINE UNITE UNITE 3 3" in chunks[0]["content"]
    assert "4 MEDLINE UNITE" in chunks[1]["content"]


def test_reflex_shaped_techniques_give_each_step_its_technique():
    # REFLEX holds 3 techniques; their two-line titles came out as chunks
    # titled "ULTRA Staple", "screw" and "and ULTRA staple" holding only
    # the figure number "1", and each "Step 1" lost its technique.
    first = "REFLEX HYBRID Implant with ULTRA Staple"
    second = "TETRA Staple with Cannulated screw"
    third = "TETRA Staple and ULTRA staple"
    text = (
        f"{_marker(8)}\n\n"
        "# REFLEX HYBRID Implant\n# with ULTRA Staple\n1\n"
        "# Step 1. Pin Template on Bone\n"
        "Provisionally pin the 1st TMT joint using a K-wire.\n"
        "NOTE: P rocedure requires Core Foot Recon Plating System. The procedure is staged.\n"
        "# Step 2. Prepare Bone for Plate Legs\nDrill the first hole with the Ø2.8mm drill bit.\n"
        f"\n{_marker(11)}\n\n"
        "# Step 8. Impact Final Construct\nUse the provided tamp.\n"
        "# Implant Removal\nIf implant removal is required, utilize the T15 driver.\n"
        f"\n{_marker(12)}\n\n"
        "# TETRA Staple with Cannulated\n# screw\n1\n"
        "# Step 1. Joint Prep\nPrep the joint using the cup and cone reamers.\n"
        "# Step 2. Provisional Pinning\nPin the 1st MTP joint with a K-wire.\n"
        f"\n{_marker(16)}\n\n"
        "# TETRA Staple\n# and ULTRA staple\n1\n"
        "# Step 1. Provisional Pinning\nPin the 1st TMT joint using a K-wire.\n"
        "# Step 3. 4-Leg Staple Bone Preparation 4\nDrill with the Ø3.0mm drill bit."
    )

    chunks = chunk_document(text)

    assert _paths(chunks) == [
        [first, "Step 1. Pin Template on Bone"],
        [first, "Step 2. Prepare Bone for Plate Legs"],
        [first, "Step 8. Impact Final Construct"],
        ["Implant Removal"],
        [second, "Step 1. Joint Prep"],
        [second, "Step 2. Provisional Pinning"],
        [third, "Step 1. Provisional Pinning"],
        [third, "Step 3. 4-Leg Staple Bone Preparation"],
    ]
    # The title's lone figure number moved into its first step, not a
    # chunk of its own; the step title's trailing figure number too.
    assert chunks[0]["content"].startswith("1\n")
    assert chunks[0]["page_number"] == 8
    assert chunks[-1]["content"].startswith("4\n")
    assert "NOTE: Procedure requires Core Foot Recon Plating System." in chunks[0]["content"]


def test_ankle_shaped_techniques_never_inherit_a_title_that_isnt_theirs():
    # Ankle Trauma: "Pilon Primary Fusion" (a product page) sits right
    # before the Lateral Fibula technique; SYNDEX starts at a bare "Step 1"
    # because its title is artwork, not text.
    pilon_body = " ".join(["The Primary Pilon option is designed for comminuted distal tibia fractures."] * 6)
    text = (
        f"{_marker(11)}\n\n"
        f"# Pilon Primary Fusion\n{pilon_body}\nMPAF601L\nMPAF601R\nmedlineunite.com 9\n"
        f"\n{_marker(12)}\n\n"
        "SURGICAL TECHNIQUE\n"
        "# Lateral Fibula Plate Surgical Technique\n"
        "# Step 1 | Exposure and reduction\nMake a straight lateral incision.\n"
        "# Optional | Plate bending\nAdditional plate contouring may be achieved with the bending irons.\n"
        "# Step 2 | Plate placement\nSelect the appropriate lateral fibula plate.\n"
        f"\n{_marker(13)}\n\n"
        "# Lateral Fibula Plate Surgical Technique,\ncontinued\n"
        "# Step 3 | Distal screw placement\nInsert the selected screw with a T15 driver.\n"
        "# Optional step 4 | Restore length\nPosition the Fibula Distractor.\n"
        f"\n{_marker(14)}\n\n"
        "SURGICAL TECHNIQUE\n"
        "# Step 1\nDrill all four cortices using the Ø3.7 mm drill bit.\n"
        "# Step 2\nInsert the passing suture through the drill bit's eyelet."
    )

    chunks = chunk_document(text)

    lateral = "Lateral Fibula Plate Surgical Technique"
    assert _paths(chunks) == [
        ["Pilon Primary Fusion"],
        [lateral, "Step 1 | Exposure and reduction"],
        [lateral, "Optional | Plate bending"],
        [lateral, "Step 2 | Plate placement"],
        [lateral, "Step 3 | Distal screw placement"],
        [lateral, "Optional step 4 | Restore length"],
        ["Step 1"],
        ["Step 2"],
    ]
    assert "continued" in chunks[4]["content"]


def test_plain_text_and_untitled_steps_keep_a_flat_path():
    assert chunk_document("First paragraph.")[0]["heading_path"] == []
    chunks = chunk_document("# Step 1. Surgical Approach\nIncise.\n# Revision Plates\nSelect a plate.")
    assert _paths(chunks) == [["Step 1. Surgical Approach"], ["Revision Plates"]]


def _mis_screw_table(row_count: int) -> str:
    rows = [
        f"MSCF30{20 + index * 2:02d} MIS FT Screw, 3.0 x {20 + index * 2}mm 3 "
        f"MSCP40{20 + index * 2:02d} MIS PT Screw, 4.0 X {20 + index * 2}mm 3"
        for index in range(row_count)
    ]
    return "\n".join(
        ["MIS Bunion and Fusion Screws", "Item No. Description Qty. Item No. Description Qty.", *rows]
    )


def test_long_tables_split_on_row_boundaries_with_the_header_repeated():
    # MIS c15/c16 ("MSCP" | " 20mm"), Lisfranc c13/c14 ("MSDL4126
    # Lisfranc" | " Screw, 4.1 X 20mm") and Ankle c66/c67 ("MPSL354"):
    # an over-budget table paragraph used to be token-windowed mid-row.
    table = _mis_screw_table(60)
    assert _tokens(table) > 800

    chunks = chunk_document(f"{_marker(12)}\n\n# Section 3\n{table}")

    assert len(chunks) > 1
    table_lines = table.split("\n")
    header = table_lines[:2]
    seen_rows: list[str] = []
    for chunk in chunks:
        lines = chunk["content"].split("\n")
        assert lines[:2] == header
        assert all(line in table_lines for line in lines)
        seen_rows.extend(lines[2:])
        assert _tokens(chunk["content"]) <= 800
        assert chunk["page_number"] == 12
        assert chunk["section_title"] == "Section 3"
    # Every row exactly once: no cut row, no row repeated by an overlap.
    assert seen_rows == table_lines[2:]


def test_a_header_starting_a_new_sub_table_isnt_preceded_by_the_old_one():
    first = _mis_screw_table(40)
    second = "\n".join(
        ["Lisfranc Screws", "Item No. Description Qty."]
        + [f"MSDL41{20 + index * 2:02d} Lisfranc Screw, 4.1 X {20 + index * 2}mm 2" for index in range(30)]
    )

    chunks = chunk_document(f"{first}\n{second}", chunk_size=400)

    assert len(chunks) > 3
    for chunk in chunks:
        lines = chunk["content"].split("\n")
        # Every piece opens with a caption and its column header, and the
        # header is the one its first row belongs to.
        assert lines[1].startswith("Item No.")
        expected_caption = "Lisfranc Screws" if lines[2].startswith("MSDL") else "MIS Bunion and Fusion Screws"
        assert lines[0] == expected_caption
        # Never closes on a caption or header with none of its rows.
        assert lines[-1] != "Lisfranc Screws"
        assert not lines[-1].startswith("Item No.")


def test_a_piece_never_ends_on_a_sub_tables_header():
    first = _mis_screw_table(40)
    second = "\n".join(
        ["Lisfranc Screws", "Item No. Description Qty."]
        + [f"MSDL41{20 + index * 2:02d} Lisfranc Screw, 4.1 X {20 + index * 2}mm 2" for index in range(30)]
    )
    text = f"{first}\n{second}"
    # Try every budget around the point where the second header lands.
    for chunk_size in range(380, 460, 7):
        for chunk in chunk_document(text, chunk_size=chunk_size):
            last = chunk["content"].split("\n")[-1]
            assert last != "Lisfranc Screws"
            assert not last.startswith("Item No.")


def test_long_prose_without_blank_lines_splits_between_lines_with_overlap():
    lines = [f"Sentence {index} about guidewire placement under fluoroscopy." for index in range(200)]
    paragraph = "\n".join(lines)

    chunks = chunk_document(paragraph, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    for chunk in chunks:
        assert all(line in lines for line in chunk["content"].split("\n"))
    first_tail = chunks[0]["content"].split("\n")[-1]
    assert chunks[1]["content"].startswith(first_tail) or first_tail in chunks[1]["content"]

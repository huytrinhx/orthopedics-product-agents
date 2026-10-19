"""Exercises backend/ingestion/text_extraction.py. .txt/.md need no fixture
beyond a plain file; PDF heading detection is exercised against a real PDF
built at test time with fpdf2 (dev-only dependency, see pyproject.toml)
rather than mocking pdfplumber, so the actual font-size heuristic runs
against a real parsed document.
"""
import pytest
from fpdf import FPDF

from ingestion.text_extraction import PAGE_MARKER_TEMPLATE, UnsupportedDocumentFormat, extract_text


def test_extract_plain_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Hello there.\n\nSecond paragraph.")

    assert extract_text(path, "notes.txt") == "Hello there.\n\nSecond paragraph."


def test_extract_markdown_is_read_as_is(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Heading\nBody text.")

    assert extract_text(path, "guide.md") == "# Heading\nBody text."


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "scan.tiff"
    path.write_bytes(b"\x00\x01")

    with pytest.raises(UnsupportedDocumentFormat):
        extract_text(path, "scan.tiff")


def _build_pdf(path) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=20)
    pdf.cell(0, 12, "Sizing Guide", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, "Use the 4.0mm screw for standard bone density.")
    pdf.set_font("Helvetica", size=20)
    pdf.cell(0, 12, "Sterilization", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, "Autoclave at 270F for 4 minutes.")
    pdf.output(str(path))


def test_extract_pdf_marks_larger_font_lines_as_headings(tmp_path):
    path = tmp_path / "brochure.pdf"
    _build_pdf(path)

    text = extract_text(path, "brochure.pdf")

    lines = text.splitlines()
    assert "# Sizing Guide" in lines
    assert "# Sterilization" in lines
    assert "Use the 4.0mm screw for standard bone density." in lines
    assert "# Use the 4.0mm screw for standard bone density." not in text


def test_extract_pdf_inserts_a_page_marker_before_each_page(tmp_path):
    path = tmp_path / "brochure.pdf"
    _build_pdf(path)

    text = extract_text(path, "brochure.pdf")

    assert PAGE_MARKER_TEMPLATE.format(page_number=1) in text
    marker_index = text.index(PAGE_MARKER_TEMPLATE.format(page_number=1))
    heading_index = text.index("# Sizing Guide")
    assert marker_index < heading_index


def test_extract_pdf_marks_a_second_page_with_a_higher_page_number(tmp_path):
    path = tmp_path / "two-page.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, "Page one body text.")
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, "Page two body text.")
    pdf.output(str(path))

    text = extract_text(path, "two-page.pdf")

    assert PAGE_MARKER_TEMPLATE.format(page_number=1) in text
    assert PAGE_MARKER_TEMPLATE.format(page_number=2) in text
    assert text.index(PAGE_MARKER_TEMPLATE.format(page_number=1)) < text.index(
        "Page one body text."
    )
    assert text.index(PAGE_MARKER_TEMPLATE.format(page_number=2)) < text.index(
        "Page two body text."
    )


def _three_column_tray_page(path) -> None:
    """Core Foot p17's shape (issue #26, chunk c57): three side-by-side
    tables, the middle column's rows a few points lower than the first's,
    and the reamers' descriptions printed a few points above their SKUs.
    Read row by row this came out as "MPSL2710 2.7 x 10 mm 4 ø16 mm" /
    "MPSN3510 3.5 x 10 mm 4 MPN20116 1"."""
    pdf = FPDF(unit="pt", format="letter")
    pdf.add_page()
    pdf.set_font("Helvetica", size=8)
    columns = [
        (30, "ø2.7 mm Polyaxial Locking Screws", [("MPSL2710", "2.7 x 10 mm", "4"), ("MPSL2712", "2.7 x 12 mm", "4")], 0, 0),
        (220, "ø3.5 mm Non-Locking Screws", [("MPSN3510", "3.5 x 10 mm", "4"), ("MPSN3512", "3.5 x 12 mm", "4")], 4, 0),
        (410, "MTP Fusion Metatarsal Reamers", [("MPN20116", "ø16 mm", "1"), ("MPN20118", "ø18 mm", "1")], 0, -4),
    ]
    for left, caption, rows, row_offset, description_offset in columns:
        pdf.text(left, 70, caption)
        pdf.text(left, 82, "Item No.")
        pdf.text(left + 50, 82, "Description")
        pdf.text(left + 150, 82, "Qty.")
        for row_index, (sku, description, quantity) in enumerate(rows):
            baseline = 100 + row_index * 18 + row_offset
            pdf.text(left, baseline, sku)
            pdf.text(left + 50, baseline + description_offset, description)
            pdf.text(left + 150, baseline, quantity)
    pdf.output(str(path))


def test_extract_pdf_reads_side_by_side_tables_one_column_at_a_time(tmp_path):
    path = tmp_path / "tray.pdf"
    _three_column_tray_page(path)

    lines = extract_text(path, "tray.pdf").splitlines()

    assert "MPSL2710 2.7 x 10 mm 4" in lines
    assert "MPSN3510 3.5 x 10 mm 4" in lines
    assert "MPN20116 ø16 mm 1" in lines
    # Each column keeps its own caption and header above its own rows.
    reamers = lines.index("MTP Fusion Metatarsal Reamers")
    assert lines[reamers + 1 : reamers + 4] == ["Item No. Description Qty.", "MPN20116 ø16 mm 1", "MPN20118 ø18 mm 1"]
    assert lines.index("MPSL2712 2.7 x 12 mm 4") < lines.index("ø3.5 mm Non-Locking Screws")


def test_extract_pdf_leaves_a_page_with_one_table_header_row_by_row(tmp_path):
    path = tmp_path / "single.pdf"
    pdf = FPDF(unit="pt", format="letter")
    pdf.add_page()
    pdf.set_font("Helvetica", size=8)
    pdf.text(30, 82, "Item No.")
    pdf.text(80, 82, "Description")
    pdf.text(30, 100, "MSN40016")
    pdf.text(80, 100, "Depth Gauge")
    pdf.text(300, 100, "Right-hand note")
    pdf.output(str(path))

    lines = extract_text(path, "single.pdf").splitlines()

    assert "MSN40016 Depth Gauge Right-hand note" in lines


def test_extract_pdf_drops_doubled_font_copies(tmp_path):
    # A shadow copy at a slightly different size came out word by word as
    # "MEDLINE MEDLINE UNITE UNITE 11 11"; a same-size copy interleaves
    # characters ("MMEEDDLLIINNEE") instead.
    path = tmp_path / "doubled.pdf"
    pdf = FPDF(unit="pt", format="letter")
    pdf.add_page()
    for top, second_size, offset in ((100, 14.3, 1.0), (140, 14, 0.3)):
        pdf.set_font("Helvetica", size=14)
        pdf.text(30, top, "MEDLINE UNITE 11")
        pdf.set_font("Helvetica", size=second_size)
        pdf.text(30 + offset, top, "MEDLINE UNITE 11")
    pdf.output(str(path))

    lines = [line.removeprefix("# ") for line in extract_text(path, "doubled.pdf").splitlines()]

    assert lines.count("MEDLINE UNITE 11") == 2

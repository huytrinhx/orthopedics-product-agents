"""Exercises backend/ingestion/text_extraction.py. .txt/.md need no fixture
beyond a plain file; PDF heading detection is exercised against a real PDF
built at test time with fpdf2 (dev-only dependency, see pyproject.toml)
rather than mocking pdfplumber, so the actual font-size heuristic runs
against a real parsed document.
"""
import pytest
from fpdf import FPDF

from ingestion.text_extraction import UnsupportedDocumentFormat, extract_text


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

"""Exercises backend/ingestion/tray_layout_extraction.py. The page-selection
heuristic is pure and LLM-free, so it's tested against a real PDF built at
test time with fpdf2 (dev-only dependency, matching test_text_extraction.py's
approach) with a real embedded image, rather than mocking pdfplumber.
`replace_tray_sections` is tested against a real Neo4j (matching
test_graph_client.py), for the idempotent-delete-then-recreate behavior that
the vision call itself is never exercised here.
"""
import uuid

import pdfplumber
from fpdf import FPDF

from ingestion.tray_layout_extraction import (
    MAX_PAGES_PER_DOCUMENT,
    _has_large_images,
    _has_tray_keyword,
    select_candidate_pages,
)
from retrieval.graph_client import get_graph_client


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _large_png_bytes(path) -> None:
    from PIL import Image

    Image.new("RGB", (400, 300), color="gray").save(path)


def _build_pdf_with_image(path, *, text: str, image_path) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    pdf.image(str(image_path), x=20, y=40, w=150, h=110)
    pdf.output(str(path))


def _build_text_only_pdf(path, *, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    pdf.output(str(path))


def test_has_tray_keyword_is_case_insensitive():
    assert _has_tray_keyword("Top Level / Bottom Level layout")
    assert _has_tray_keyword("TRAY LAYOUT overview")
    assert not _has_tray_keyword("Screw sizing chart")


def test_select_candidate_pages_requires_both_signals(tmp_path):
    image_path = tmp_path / "photo.png"
    _large_png_bytes(image_path)

    tray_page_with_image = tmp_path / "tray.pdf"
    _build_pdf_with_image(tray_page_with_image, text="Tray Layout\nTop Level", image_path=image_path)
    candidates = select_candidate_pages(tray_page_with_image)
    assert [c["page_number"] for c in candidates] == [1]
    assert "Tray Layout" in candidates[0]["text"]

    image_no_keyword = tmp_path / "image_no_keyword.pdf"
    _build_pdf_with_image(image_no_keyword, text="Inventory list, page 3", image_path=image_path)
    assert select_candidate_pages(image_no_keyword) == []

    keyword_no_image = tmp_path / "keyword_no_image.pdf"
    _build_text_only_pdf(keyword_no_image, text="Tray Layout\nTop Level")
    assert select_candidate_pages(keyword_no_image) == []


def test_select_candidate_pages_caps_at_max_pages(tmp_path):
    image_path = tmp_path / "photo.png"
    _large_png_bytes(image_path)

    pdf = FPDF()
    for _ in range(MAX_PAGES_PER_DOCUMENT + 5):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, "Tray Layout\nTop Level")
        pdf.image(str(image_path), x=20, y=40, w=150, h=110)
    path = tmp_path / "many_pages.pdf"
    pdf.output(str(path))

    candidates = select_candidate_pages(path)
    assert len(candidates) == MAX_PAGES_PER_DOCUMENT


def test_has_large_images_ignores_small_images(tmp_path):
    small_image_path = tmp_path / "icon.png"
    from PIL import Image

    Image.new("RGB", (40, 40), color="gray").save(small_image_path)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, "Tray Layout")
    pdf.image(str(small_image_path), x=20, y=40, w=20, h=20)
    path = tmp_path / "small_icon.pdf"
    pdf.output(str(path))

    with pdfplumber.open(path) as doc:
        page = doc.pages[0]
        assert not _has_large_images(page)


async def test_replace_tray_sections_is_idempotent_on_reingestion():
    client = get_graph_client()
    await client.ensure_constraints()
    tray = _unique("TRAY")
    family = _unique("FAMILY")
    sku_a = _unique("SKU-A")
    sku_b = _unique("SKU-B")
    doc_id = _unique("DOC")

    await client.upsert_product_family(family)
    await client.upsert_tray(tray, family)
    await client.upsert_part(sku_a, tray)
    await client.upsert_part(sku_b, tray)

    await client.replace_tray_sections(
        doc_id,
        [{"tray": tray, "level": "Top Level", "region": "Left Side", "skus": [sku_a, sku_b]}],
    )

    async with client._driver.session() as session:
        result = await session.run(
            "MATCH (p:Part)-[r:LOCATED_IN {document_id: $doc_id}]->(s:TraySection) "
            "RETURN p.sku AS sku, s.tray AS tray, s.level AS level, s.region AS region",
            doc_id=doc_id,
        )
        rows = [dict(record) async for record in result]
    assert {r["sku"] for r in rows} == {sku_a, sku_b}
    assert all(r["tray"] == tray and r["level"] == "Top Level" and r["region"] == "Left Side" for r in rows)

    # Re-ingestion with a corrected grouping (sku_a moved out) should leave
    # no stale edge from the first run behind.
    await client.replace_tray_sections(
        doc_id,
        [{"tray": tray, "level": "Top Level", "region": "Right Side", "skus": [sku_b]}],
    )

    async with client._driver.session() as session:
        result = await session.run(
            "MATCH (p:Part)-[r:LOCATED_IN {document_id: $doc_id}]->(s:TraySection) "
            "RETURN p.sku AS sku, s.region AS region",
            doc_id=doc_id,
        )
        rows = [dict(record) async for record in result]
    assert rows == [{"sku": sku_b, "region": "Right Side"}]


async def test_replace_tray_sections_does_not_touch_other_documents_edges():
    client = get_graph_client()
    await client.ensure_constraints()
    tray = _unique("TRAY")
    family = _unique("FAMILY")
    sku = _unique("SKU")
    doc_id_a = _unique("DOC-A")
    doc_id_b = _unique("DOC-B")

    await client.upsert_product_family(family)
    await client.upsert_tray(tray, family)
    await client.upsert_part(sku, tray)

    await client.replace_tray_sections(
        doc_id_a, [{"tray": tray, "level": "", "region": "Middle", "skus": [sku]}]
    )
    await client.replace_tray_sections(
        doc_id_b, [{"tray": tray, "level": "", "region": "Middle", "skus": [sku]}]
    )

    async with client._driver.session() as session:
        result = await session.run(
            "MATCH (:Part {sku: $sku})-[r:LOCATED_IN]->(:TraySection) RETURN r.document_id AS doc_id",
            sku=sku,
        )
        doc_ids = {record["doc_id"] async for record in result}
    assert doc_ids == {doc_id_a, doc_id_b}

    # Re-running doc A alone must not delete doc B's edge.
    await client.replace_tray_sections(doc_id_a, [])

    async with client._driver.session() as session:
        result = await session.run(
            "MATCH (:Part {sku: $sku})-[r:LOCATED_IN]->(:TraySection) RETURN r.document_id AS doc_id",
            sku=sku,
        )
        doc_ids = {record["doc_id"] async for record in result}
    assert doc_ids == {doc_id_b}

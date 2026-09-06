"""Ticket 25: extracts which known Part SKUs sit in which physical tray
region from tray photos embedded in ingested documents, writing
`Part -[:LOCATED_IN]-> TraySection` edges into the graph (backend/retrieval/
graph_client.py) — the one piece of ticket 07's schema left deliberately
unpopulated ("shape only... until tray-overhead-guide extraction exists").

This is a third, independent ingestion leg (backend/documents/service.py),
sibling to backend/ingestion/entity_extraction.py (graph leg, prose facts)
and backend/ingestion/chunking.py/embedding.py (vector leg) — triggered from
the same place, but never blocks or fails either of them (best-effort: a
document with no tray photo, an untagged system, or a non-PDF file is a
silent no-op, not an error).

No document-type gating: `document_types` are free-text admin tags (ticket
05), not a fixed enum, so trusting an exact tag string would silently skip
a mistagged or oddly-named document. Instead, `_select_candidate_pages`
below runs a cheap, LLM-free two-signal content heuristic on every ingested
PDF regardless of its tag -- only a page with both large embedded images
*and* tray/level language in its own text goes to vision. Found live
(2026-09-05) against a real tray-design slide (Medline Unite MIS Foot Recon
System - Customer Presentation.pdf, page 8) that both signals hold there
and a plain text/inventory-table page has neither, so the combination
should stay precise without needing per-document-type tuning.

Extraction stays coarse, matching ticket 07's original scope: a tray name +
level + region label (e.g. "Top Level", "Left Side"), never pixel
coordinates -- vision's job is reading a region's own printed labels (exact
diameter/thread codes baked into the tray's photo, which the page's caption
text alone doesn't repeat) well enough to sort known SKUs into the right
region, not producing a spatial map. Reuses entity_extraction.py's
established convention: the model is given the real "known parts"/"known
trays" lists for the document's tagged system and may only reference names
from them -- anything else is dropped here (and again, defensively, at the
graph write layer) rather than merged as a fuzzy match or a new node.
"""
import base64
import io
import logging
from pathlib import Path

import pdfplumber
from pydantic import BaseModel, Field

from config.llm_clients import get_chat_model
from retrieval.graph_client import get_graph_client

logger = logging.getLogger(__name__)

# Neo4j round-trips and vision calls are both more expensive than a text
# extraction call -- capped so a malformed or unusually large PDF (lots of
# image-heavy pages) can't fire an unbounded number of vision calls. Every
# real document seen so far has at most one tray-photo page, so this is
# pure headroom, not a real expected ceiling.
MAX_PAGES_PER_DOCUMENT = 20

# A line's own text is the caption's category names ("Screw Prep
# Instruments (Middle)") and the "Top Level"/"Bottom Level" headers -- these
# are real PDF text elements, not baked into the tray photo's pixels, so
# pdfplumber's plain text layer already has them for free.
_TRAY_KEYWORDS = (
    "top level",
    "bottom level",
    "tray design",
    "left side",
    "right side",
    "tray layout",
    "instrument tray",
)

# PDF points (1/72 inch). The real tray photos on the first confirmed
# example page are ~476x218 and ~479x218 points; a page logo/icon on that
# same page is ~126x33 -- comfortably below this on both dimensions.
# Requiring *both* dimensions above the threshold (not just area) rules out
# a single long, thin banner image that happens to have a large area.
_MIN_IMAGE_DIMENSION = 150
# Originally 2 (the first example page has two separate top/bottom-level
# photos) -- lowered to 1 after finding a second real example live
# (Medline Unite MIS Foot Recon System - System Overview and Surgical
# Technique.pdf, page 14, titled "Tray Layout") where both tray levels are
# combined into a single wide photo with numbered section call-outs. One
# large image plus the keyword signal is still a precise combination in
# practice -- an unrelated page with one large photo essentially never also
# contains "Tray Layout"/"Top Level"/etc. verbatim.
_MIN_LARGE_IMAGES = 1

_SYSTEM_PROMPT = """You read one page from orthopedic product documentation \
that shows a photo of a physical instrument/implant tray, divided into \
labeled regions (e.g. "Top Level" vs "Bottom Level" as separate physical \
levels, and "Left Side" / "Middle" / "Right Side" as regions within a \
level).

For each labeled region actually visible on this page, identify which of \
the known parts (given below) are physically located in it. Use both the \
page's own caption text (category names like "Screw Prep Instruments") and \
anything printed directly on the tray in the photo itself (exact sizes, \
diameters, thread-type codes stenciled onto the tray) to decide which \
specific known SKUs belong in each region -- the photo's own printed \
labels are often more precise than the caption text alone.

You may ONLY reference a tray name that appears in the "Known trays" list \
below, exactly as given, and a SKU that appears in the "Known parts" list \
below, exactly as given. If you cannot confidently match a region's \
contents to specific known SKUs, leave that region's "skus" list empty (or \
partial) rather than guessing -- and return no groups at all if this page \
doesn't actually show a tray layout. If the tray has no separate physical \
levels, leave "level" as an empty string and use "region" alone.
"""


class _TraySectionGroup(BaseModel):
    tray: str = Field(description="Tray name, copied exactly from the known trays list")
    level: str = Field(
        default="",
        description=(
            "Physical level within the tray, e.g. 'Top Level' or 'Bottom Level' -- "
            "empty string if the tray has no separate levels"
        ),
    )
    region: str = Field(description="Region within that level, e.g. 'Left Side', 'Middle', 'Right Side'")
    skus: list[str] = Field(
        default_factory=list, description="Known SKUs visibly located in this region"
    )


class _TrayLayoutExtraction(BaseModel):
    groups: list[_TraySectionGroup] = Field(default_factory=list)


def _image_dimensions_points(image: dict) -> tuple[float, float]:
    return image["x1"] - image["x0"], image["y1"] - image["y0"]


def _has_large_images(page, min_count: int = _MIN_LARGE_IMAGES) -> bool:
    large = [
        img
        for img in page.images
        if all(dim >= _MIN_IMAGE_DIMENSION for dim in _image_dimensions_points(img))
    ]
    return len(large) >= min_count


def _has_tray_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _TRAY_KEYWORDS)


def select_candidate_pages(storage_path: str | Path) -> list[dict]:
    """The cheap, LLM-free heuristic: a page qualifies only if it has
    multiple large embedded images *and* its own text mentions tray/level
    language. Pure function of the PDF's own content (pdfplumber metadata +
    text already extracted for free) -- no network call, safe to run on
    every ingested PDF regardless of its document-type tag. Returns
    `{"page_number": int, "text": str}` for each qualifying page (1-indexed,
    matching how a human would refer to a page), capped at
    MAX_PAGES_PER_DOCUMENT.
    """
    candidates: list[dict] = []
    with pdfplumber.open(storage_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if len(candidates) >= MAX_PAGES_PER_DOCUMENT:
                logger.warning(
                    "tray_layout_extraction: hit MAX_PAGES_PER_DOCUMENT (%d) on %s, "
                    "skipping the rest of the document",
                    MAX_PAGES_PER_DOCUMENT,
                    storage_path,
                )
                break
            text = page.extract_text() or ""
            if _has_large_images(page) and _has_tray_keyword(text):
                candidates.append({"page_number": i + 1, "text": text})
    return candidates


def _render_page_png_b64(storage_path: str | Path, page_number: int) -> str:
    with pdfplumber.open(storage_path) as pdf:
        page = pdf.pages[page_number - 1]
        image = page.to_image(resolution=150).original
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def _extract_page_groups(
    image_b64: str, caption_text: str, *, known_parts: list[dict], known_trays: list[str]
) -> list[_TraySectionGroup]:
    catalog_listing = "\n".join(f"- {part['sku']}: {part['description']}" for part in known_parts)
    trays_listing = "\n".join(f"- {tray}" for tray in known_trays)

    model = get_chat_model().with_structured_output(_TrayLayoutExtraction)
    result = await model.ainvoke(
        [
            ("system", _SYSTEM_PROMPT),
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": (
                            f"Known trays:\n{trays_listing}\n\n"
                            f"Known parts:\n{catalog_listing}\n\n"
                            f"This page's own text (for context; the image is the "
                            f"authoritative source for exact per-region labels):\n{caption_text}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            ),
        ]
    )
    return result.groups


async def extract_tray_layout(
    document_id: str,
    storage_path: str | Path,
    filename: str,
    *,
    system: str | None,
) -> int:
    """Entry point called from backend/documents/service.py, after the graph
    leg (ingest_document) has run so list_parts_for_family/
    list_trays_for_family already reflect this document's system. Returns
    the number of Part -[:LOCATED_IN]-> TraySection edges written (0 for
    every no-op case: no system, no known parts/trays yet, not a PDF, or no
    page passes the heuristic). Never raises -- a bug in vision extraction
    shouldn't take down the other two ingestion legs; the caller doesn't
    need to wrap this in its own try/except.
    """
    try:
        if not system or Path(filename).suffix.lower() != ".pdf":
            return 0

        client = get_graph_client()
        known_parts = await client.list_parts_for_family(system)
        known_trays = await client.list_trays_for_family(system)
        if not known_parts or not known_trays:
            return 0

        candidates = select_candidate_pages(storage_path)
        if not candidates:
            return 0

        known_skus = {part["sku"] for part in known_parts}
        known_tray_set = set(known_trays)

        written = 0
        all_valid_groups: list[_TraySectionGroup] = []
        for candidate in candidates:
            image_b64 = _render_page_png_b64(storage_path, candidate["page_number"])
            groups = await _extract_page_groups(
                image_b64, candidate["text"], known_parts=known_parts, known_trays=known_trays
            )
            for group in groups:
                if group.tray not in known_tray_set:
                    continue
                valid_skus = [sku for sku in group.skus if sku in known_skus]
                if not valid_skus:
                    continue
                all_valid_groups.append(
                    _TraySectionGroup(
                        tray=group.tray, level=group.level, region=group.region, skus=valid_skus
                    )
                )

        if not all_valid_groups:
            return 0

        await client.replace_tray_sections(
            document_id,
            [
                {"tray": g.tray, "level": g.level, "region": g.region, "skus": g.skus}
                for g in all_valid_groups
            ],
        )
        written = sum(len(g.skus) for g in all_valid_groups)
        return written
    except Exception:
        logger.exception(
            "tray_layout_extraction failed for document_id=%s (%s) -- other ingestion "
            "legs are unaffected",
            document_id,
            filename,
        )
        return 0

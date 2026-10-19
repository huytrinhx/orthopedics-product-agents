"""The per-document SKU index (issue #27): every SKU a document's vector
chunks mention, with the chunk it's in, found without a model call.

`scan_chunks` is the whole scan and is pure, so it's tested on a fixture
catalog. It runs these passes over each chunk:

1. A regex pass for SKU-shaped tokens (`_CANDIDATE_PATTERN`). Each one is
   validated against the **whole** catalog, not just the document's family,
   because documents name other families' parts: REFLEX HYBRID items in the
   REFLEX document, Distal Tibia items in Ankle Trauma.
2. An exact-match pass for the catalog SKUs that pattern can't produce:
   those with no digit (`SNAPWIRE`), a hyphen (`TS-4515`), a leading digit,
   or fewer than 6 characters (`MA830`).
3. Series and side alternatives. "MSFB0xxx series" resolves to every
   catalog SKU with that prefix and length; "MPP1101L/R" adds MPP1101R.
4. A token that looks like a SKU but isn't in the catalog is kept and
   flagged (`not_in_catalog`) for review: a typo (`MPP202L`), a cut-off
   (`MPSL354`), or a part the catalog lacks (`MSN20003`). "Looks like a
   SKU" means it starts with the same three characters as some catalog
   SKU. That drops literature and marketing codes (`LIT350`,
   `MKT19W6341649`).

Descriptions are read from the catalog when the index is read, never taken
from the document: PDF text drops digits ("Ø1.2 x 50mm" for a 1.2 x 150mm
guidepin) and mis-pairs multi-column rows. The row's line is kept only as
provenance.
"""
import bisect
import logging
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from document_skus import repository
from document_skus.models import DocumentSkuOut
from document_skus.repository import DocumentSkuRow
from retrieval.graph_client import get_graph_client

logger = logging.getLogger(__name__)

EXACT = "exact"
SERIES = "series"
NOT_IN_CATALOG = "not_in_catalog"

# Two or more capitals, then capitals and digits with at least one digit,
# not touching another letter, digit or hyphen. The negative lookahead
# matters: "MSFB0xxx" must not yield "MSFB0", and "MPP1101L/R" still yields
# "MPP1101L". Covers the `NS` (MST41515NS), side (MPP1101L) and trailing
# digit (MSTTAMP1) shapes as whole tokens. Same idea as
# agents/fact_check.py's _SKU_PATTERN, minus the hyphen.
_CANDIDATE_PATTERN = re.compile(r"(?<![A-Za-z0-9-])[A-Z]{2,}[A-Z0-9]*\d[A-Z0-9]*(?![A-Za-z0-9-])")
_MIN_CANDIDATE_LENGTH = 6
# A prefix, then a run of wildcard x's: "MSFB0xxx". Lowercase x's, or two
# or more capitals (catalog SKUs contain X, e.g. MSTX2600NS, never XX).
_SERIES_PATTERN = re.compile(r"(?<![A-Za-z0-9-])([A-Z]{2,}[A-Z0-9]*?)(x{2,}|X{2,})(?![A-Za-z0-9-])")
# A sided SKU followed by its other side(s): "MPP1101L/R".
_SIDE_ALTERNATIVES_PATTERN = re.compile(
    r"(?<![A-Za-z0-9-])([A-Z]{2,}[A-Z0-9]*\d[A-Z0-9]*?)([LRU])((?:/[LRU])+)(?![A-Za-z0-9-])"
)
_SKU_SHAPE_PREFIX_LENGTH = 3

# A table's column-header row ("Item No. Description Qty."). The table
# heading is the line just above it.
_COLUMN_HEADER_PATTERN = re.compile(
    r"\bitem\s+no\b|\bpart\s+(?:no|number)\b|\bcatalog\s+(?:no|number)\b|\bdescription\s+qty\b",
    re.IGNORECASE,
)
# Multi-column tables are flattened line by line, so the other column's
# rows can sit between a heading and its column header.
_MAX_ROWS_BETWEEN_HEADING_AND_HEADER = 2
_MAX_TABLE_HEADING_LENGTH = 120
_MAX_ROW_TEXT_LENGTH = 500


@dataclass(frozen=True)
class CatalogPart:
    sku: str
    description: str | None
    families: frozenset[str]


class SkuCatalog:
    """The whole Part catalog, indexed for the scan."""

    def __init__(self, parts: Iterable[CatalogPart]):
        self.parts_by_sku = {part.sku: part for part in parts}
        self.sku_shape_prefixes = {
            sku[:_SKU_SHAPE_PREFIX_LENGTH]
            for sku in self.parts_by_sku
            if len(sku) >= _SKU_SHAPE_PREFIX_LENGTH
        }
        exact_pass_skus = sorted(
            (sku for sku in self.parts_by_sku if not _is_candidate_shaped(sku)),
            key=len,
            reverse=True,
        )
        self.exact_pass_pattern = (
            re.compile(
                r"(?<![A-Za-z0-9-])(?:"
                + "|".join(re.escape(sku) for sku in exact_pass_skus)
                + r")(?![A-Za-z0-9-])"
            )
            if exact_pass_skus
            else None
        )

    def series_members(self, prefix: str, length: int) -> list[str]:
        return sorted(
            sku for sku in self.parts_by_sku if len(sku) == length and sku.startswith(prefix)
        )

    def looks_like_sku(self, token: str) -> bool:
        return token[:_SKU_SHAPE_PREFIX_LENGTH] in self.sku_shape_prefixes

    def family_part_count(self, document_family: str | None) -> int:
        """How many catalog parts belong to the document's family."""
        if document_family is None:
            return 0
        return sum(1 for part in self.parts_by_sku.values() if document_family in part.families)


def _is_candidate_shaped(sku: str) -> bool:
    return len(sku) >= _MIN_CANDIDATE_LENGTH and _CANDIDATE_PATTERN.fullmatch(sku) is not None


class _ChunkLines:
    """A chunk's lines, to map a match position to the line it's on."""

    def __init__(self, content: str):
        self.lines = content.split("\n")
        self.line_starts: list[int] = []
        position = 0
        for line in self.lines:
            self.line_starts.append(position)
            position += len(line) + 1

    def line_index(self, position: int) -> int:
        return bisect.bisect_right(self.line_starts, position) - 1

    def row_text(self, line_index: int) -> str:
        return self.lines[line_index].strip()[:_MAX_ROW_TEXT_LENGTH]

    def table_heading(self, line_index: int) -> str | None:
        """The line above the nearest column-header row above this line,
        skipping header rows and up to two rows of the other column.
        Best-effort: None when there's no header above it in this chunk or
        the line found doesn't read as a heading."""
        header_index = next(
            (
                index
                for index in range(line_index - 1, -1, -1)
                if _COLUMN_HEADER_PATTERN.search(self.lines[index])
            ),
            None,
        )
        if header_index is None:
            return None
        index = header_index - 1
        while index >= 0 and _COLUMN_HEADER_PATTERN.search(self.lines[index]):
            index -= 1
        rows_skipped = 0
        while index >= 0:
            line = self.lines[index].strip()
            index -= 1
            if not line:
                continue
            if _CANDIDATE_PATTERN.search(line):
                rows_skipped += 1
                if rows_skipped > _MAX_ROWS_BETWEEN_HEADING_AND_HEADER:
                    return None
                continue
            if re.search(r"[A-Za-z]{3,}", line) and len(line) <= _MAX_TABLE_HEADING_LENGTH:
                return line
            return None
        return None


def scan_chunks(
    chunks: Iterable[tuple[int, str]], catalog: SkuCatalog, document_family: str | None
) -> list[DocumentSkuRow]:
    """Every SKU mention in the given (chunk_index, content) pairs. One row
    per (chunk, SKU, match kind, line): a SKU listed twice in one chunk with
    different text keeps both lines, which is how a document contradicting
    itself shows up."""
    rows: list[DocumentSkuRow] = []
    seen: set[tuple[int, str, str, str]] = set()

    def add(
        chunk_index: int, sku: str, mention: str, match_kind: str, lines: _ChunkLines, position: int
    ) -> None:
        line_index = lines.line_index(position)
        row_text = lines.row_text(line_index)
        key = (chunk_index, sku, match_kind, row_text)
        if key in seen:
            return
        seen.add(key)
        part = catalog.parts_by_sku.get(sku)
        rows.append(
            DocumentSkuRow(
                chunk_index=chunk_index,
                sku=sku,
                mention=mention,
                match_kind=match_kind,
                in_document_family=(
                    None if part is None else (document_family or "") in part.families
                ),
                table_heading=lines.table_heading(line_index),
                row_text=row_text,
            )
        )

    for chunk_index, content in chunks:
        lines = _ChunkLines(content)

        for match in _CANDIDATE_PATTERN.finditer(content):
            token = match.group(0)
            if len(token) < _MIN_CANDIDATE_LENGTH:
                continue
            if token in catalog.parts_by_sku:
                add(chunk_index, token, token, EXACT, lines, match.start())
            elif catalog.looks_like_sku(token):
                add(chunk_index, token, token, NOT_IN_CATALOG, lines, match.start())

        if catalog.exact_pass_pattern is not None:
            for match in catalog.exact_pass_pattern.finditer(content):
                add(chunk_index, match.group(0), match.group(0), EXACT, lines, match.start())

        for match in _SERIES_PATTERN.finditer(content):
            mention = match.group(0)
            if mention in catalog.parts_by_sku:
                continue
            members = catalog.series_members(match.group(1), len(mention))
            for sku in members:
                add(chunk_index, sku, mention, SERIES, lines, match.start())
            if not members and catalog.looks_like_sku(mention):
                add(chunk_index, mention, mention, NOT_IN_CATALOG, lines, match.start())

        for match in _SIDE_ALTERNATIVES_PATTERN.finditer(content):
            stem, alternatives = match.group(1), match.group(3)
            for side in alternatives.split("/")[1:]:
                sku = stem + side
                if sku in catalog.parts_by_sku:
                    add(chunk_index, sku, match.group(0), SERIES, lines, match.start())
                elif catalog.looks_like_sku(sku):
                    add(chunk_index, sku, match.group(0), NOT_IN_CATALOG, lines, match.start())

    return rows


async def load_catalog() -> SkuCatalog:
    parts = await get_graph_client().list_catalog_parts()
    return SkuCatalog(
        CatalogPart(
            sku=part["sku"],
            description=part.get("description"),
            families=frozenset(part.get("families") or []),
        )
        for part in parts
        if part.get("sku")
    )


async def index_document_skus(document_id: uuid.UUID | str, *, product_family: str | None) -> None:
    """Rebuilds one document's SKU index from the chunks it has stored now.
    `product_family` is the document's catalog name; it only decides
    `in_document_family`, so a document whose family has no parts is still
    indexed (every match is then out of family), and that's logged."""
    chunks = await repository.list_chunk_texts(document_id)
    catalog = await load_catalog()
    if not catalog.parts_by_sku:
        logger.warning(
            "document_sku_index_empty_catalog document_id=%s: no catalog parts to match against",
            document_id,
            extra={"event": "document_sku_index_empty_catalog", "document_id": str(document_id)},
        )
    elif catalog.family_part_count(product_family) == 0:
        logger.warning(
            "document_sku_index_family_has_no_parts document_id=%s product_family=%r: "
            "every SKU found will be marked out of family",
            document_id,
            product_family,
            extra={
                "event": "document_sku_index_family_has_no_parts",
                "document_id": str(document_id),
                "product_family": product_family,
            },
        )

    rows = scan_chunks(chunks, catalog, product_family)
    await repository.replace_document_skus(document_id, rows)

    counts = {
        "in_family": sum(1 for row in rows if row.in_document_family is True),
        "other_family": sum(1 for row in rows if row.in_document_family is False),
        "not_in_catalog": sum(1 for row in rows if row.match_kind == NOT_IN_CATALOG),
    }
    flagged = sorted({row.sku for row in rows if row.match_kind == NOT_IN_CATALOG})
    logger.info(
        "document_sku_index document_id=%s chunks=%d rows=%d in_family=%d other_family=%d "
        "not_in_catalog=%d flagged=%s",
        document_id,
        len(chunks),
        len(rows),
        counts["in_family"],
        counts["other_family"],
        counts["not_in_catalog"],
        ",".join(flagged),
        extra={
            "event": "document_sku_index",
            "document_id": str(document_id),
            "chunks": len(chunks),
            **counts,
            "flagged": flagged,
        },
    )


async def list_document_skus(document_id: uuid.UUID | str) -> list[DocumentSkuOut]:
    """A document's index with each catalog SKU's description read from the
    catalog now."""
    records = await repository.list_document_skus(document_id)
    catalog_skus = sorted({record.sku for record in records if record.match_kind != NOT_IN_CATALOG})
    descriptions = {
        part["sku"]: part.get("description")
        for part in (
            await get_graph_client().list_catalog_parts(skus=catalog_skus) if catalog_skus else []
        )
    }
    return [
        DocumentSkuOut(
            sku=record.sku,
            mention=record.mention,
            match_kind=record.match_kind,
            in_document_family=record.in_document_family,
            description=descriptions.get(record.sku),
            chunk_index=record.chunk_index,
            citation=f"{document_id}#{record.chunk_index}",
            page_number=record.page_number,
            section_title=record.section_title,
            table_heading=record.table_heading,
            row_text=record.row_text,
        )
        for record in records
    ]

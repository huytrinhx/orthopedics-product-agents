"""Issue #27's per-document SKU index (backend/document_skus/).

The scan tests run `scan_chunks` on a small fixture catalog, with snippets
taken from production's Surgical Technique chunk text. The storage tests
run against real Postgres, like test_vector_store.py, with the catalog
read swapped for the same fixture so they don't depend on what Neo4j holds.
"""
import uuid

from auth.repository import create_user
from document_skus import service
from document_skus.service import CatalogPart, SkuCatalog, scan_chunks
from documents.repository import create_document, delete_document
from retrieval.vector_store import get_vector_store

MIS = "MIS - Foot Recon System"
REFLEX = "REFLEX Foot Recon Systems"
HYBRID = "REFLEX HYBRID"
ANKLE = "Ankle Fracture Plating Sytem"
DISTAL_TIBIA = "Distal Tibia Plating System"

_FIXTURE_PARTS = [
    CatalogPart("MSCF3020", "SCREW, MIS FT, 3.0 X 20MM", frozenset({MIS})),
    CatalogPart("MSG12150", "GUIDEPIN, MIS 3.0, 1.2 X 150MM", frozenset({MIS})),
    CatalogPart("SNAPWIRE", "GUIDEWIRE, 1.1 X 50MM", frozenset({MIS})),
    CatalogPart("MPP1101L", "PLATE, REVISION, LEFT", frozenset({MIS})),
    CatalogPart("MPP1101R", "PLATE, REVISION, RIGHT", frozenset({MIS})),
    CatalogPart("MST41515NS", "STAPLE, 15 X 15MM, NON-STERILE", frozenset({REFLEX})),
    CatalogPart("MSTTAMP1", "TAMP", frozenset({REFLEX})),
    CatalogPart("MPST100L", "LAPIDUS IMPLANT, LEFT", frozenset({HYBRID})),
    CatalogPart("MPN52012", "SMALL JOINT ARTHROTOME", frozenset({"Ankle Fusion"})),
    CatalogPart("MPPA202L", "HOOK PLATE, MEDIUM, LEFT", frozenset({ANKLE})),
    CatalogPart("MPPA010U", "LATERAL FIBULA PLATE, 10 HOLE", frozenset({ANKLE})),
    CatalogPart("MPPA003U", "ONE-THIRD TUBULAR PLATE, 3 HOLE", frozenset({ANKLE})),
    CatalogPart("MPSL3540", "SCREW, LOCKING, 3.5 X 40MM", frozenset({ANKLE, "Core Foot Plating System"})),
    CatalogPart("MSN20014", "COUNTERSINK", frozenset({ANKLE})),
    CatalogPart("MSFB0065", "IM FIBULA IMPLANT, 65MM", frozenset({DISTAL_TIBIA})),
    CatalogPart("MSFB0100", "IM FIBULA IMPLANT, 100MM", frozenset({DISTAL_TIBIA})),
    CatalogPart("MSFB0150", "IM FIBULA IMPLANT, 150MM", frozenset({DISTAL_TIBIA})),
    CatalogPart("MSFB1000", "NOT IN THE 0xxx SERIES", frozenset({DISTAL_TIBIA})),
    CatalogPart("TS-4515", "HYPHENATED SKU", frozenset({ANKLE})),
    CatalogPart("MA830", "FIVE-CHARACTER SKU", frozenset({ANKLE})),
]
CATALOG = SkuCatalog(_FIXTURE_PARTS)


def _rows(content: str, family: str | None = MIS, chunk_index: int = 0):
    return scan_chunks([(chunk_index, content)], CATALOG, family)


def _by_sku(rows):
    return {row.sku: row for row in rows}


def test_whole_tokens_match_including_ns_side_and_trailing_digit_suffixes():
    rows = _by_sku(
        _rows(
            "MST41515NS Staple 15 x 15mm\nMSTTAMP1 Tamp 1\nMPP1101L Revision Plate, Left\n"
            "MSCF3020 MIS FT Screw, 3.0 x 20mm 3",
            family=REFLEX,
        )
    )

    assert set(rows) == {"MST41515NS", "MSTTAMP1", "MPP1101L", "MSCF3020"}
    assert {row.match_kind for row in rows.values()} == {"exact"}
    assert rows["MST41515NS"].in_document_family is True
    assert rows["MSTTAMP1"].in_document_family is True
    assert rows["MPP1101L"].in_document_family is False


def test_catalog_skus_the_regex_cant_produce_are_found_by_exact_match():
    rows = _by_sku(
        _rows(
            "SNAPWIRE Ø1.1 x 50mm Guidewire 6\nTS-4515 Template\nMA830 Handle\n"
            "SNAPWIRES and SnapWire aren't SKUs"
        )
    )

    assert set(rows) == {"SNAPWIRE", "TS-4515", "MA830"}
    assert rows["SNAPWIRE"].row_text == "SNAPWIRE Ø1.1 x 50mm Guidewire 6"
    assert rows["SNAPWIRE"].in_document_family is True


def test_other_families_skus_are_matched_and_marked_out_of_family():
    rows = _by_sku(
        _rows("MPST100L Lapidus Implant, Left 1\nMPN52012 Small Joint Arthrotome 1", family=REFLEX)
    )

    assert rows["MPST100L"].match_kind == "exact"
    assert rows["MPST100L"].in_document_family is False
    assert rows["MPN52012"].in_document_family is False


def test_a_part_in_several_families_is_in_family_for_any_of_them():
    assert _rows("MPSL3540 3.5 x 40 mm 4", family="Core Foot Plating System")[0].in_document_family


def test_sku_shaped_tokens_not_in_the_catalog_are_kept_and_flagged():
    rows = _by_sku(
        _rows(
            "MPP202L Medium Left 1\nMPPA010V 10 Hole 1\nMPPADD3U MPPA008U\n"
            "MPSN2730 2.7 x 30 mm 2\nMPSL354\nDrill Guide MSN20003 1",
            family=ANKLE,
        )
    )

    for token in ("MPP202L", "MPPA010V", "MPPADD3U", "MPSL354", "MSN20003"):
        assert rows[token].match_kind == "not_in_catalog", token
        assert rows[token].in_document_family is None
        assert rows[token].mention == token


def test_literature_and_marketing_codes_are_not_flagged():
    rows = _rows("of Medline Industries, LP. MKT19W7262953 / LIT350 / FXSRY9010 / ISO13485")

    assert rows == []


def test_a_series_resolves_to_every_catalog_sku_with_its_prefix_and_length():
    rows = _rows("MPDT101R MPDT104R MPDT107R\nMSFB0xxx series\n6-HOLE (96mm)", family=ANKLE)

    series_rows = [row for row in rows if row.match_kind == "series"]
    assert [row.sku for row in series_rows] == ["MSFB0065", "MSFB0100", "MSFB0150"]
    assert {row.mention for row in series_rows} == {"MSFB0xxx"}
    assert {row.row_text for row in series_rows} == {"MSFB0xxx series"}
    assert all(row.in_document_family is False for row in series_rows)
    # The prefix alone isn't a candidate token.
    assert "MSFB0" not in {row.sku for row in rows}


def test_a_series_that_matches_nothing_is_flagged():
    rows = _rows("MSFZxxx series")

    assert [(row.sku, row.match_kind) for row in rows] == [("MSFZxxx", "not_in_catalog")]


def test_a_side_alternative_adds_the_other_sides_sku():
    rows = _rows("MPP1101L/R Revision Plate")

    assert {(row.sku, row.match_kind, row.mention) for row in rows} == {
        ("MPP1101L", "exact", "MPP1101L"),
        ("MPP1101R", "series", "MPP1101L/R"),
    }


def test_table_heading_is_the_line_above_the_column_header():
    content = (
        "Screw Prep Instruments\nMIS Bunion Instruments\nItem No. Description Qty.\n"
        "Item No. Description Qty.\nMSG12150 Ø1.2 x 50mm Guidepin 10\nMSCF3020 Screw 3"
    )

    rows = _by_sku(_rows(content))

    assert rows["MSG12150"].table_heading == "MIS Bunion Instruments"
    assert rows["MSCF3020"].table_heading == "MIS Bunion Instruments"
    # The document's own (wrong) text is provenance only.
    assert rows["MSG12150"].row_text == "MSG12150 Ø1.2 x 50mm Guidepin 10"


def test_table_heading_skips_the_other_columns_rows_and_is_none_without_a_header():
    interleaved = (
        "Straight Fibula Plates\nMPPA202L Standard, Left 2 MPPA010U Extra Small 2\n"
        "Item No. Description Qty.\nMPPA003U Small 2"
    )
    assert _by_sku(_rows(interleaved, family=ANKLE))["MPPA003U"].table_heading == (
        "Straight Fibula Plates"
    )
    assert _rows("Insert the MSCF3020 screw over the wire.")[0].table_heading is None


def test_one_row_per_chunk_sku_and_line():
    rows = scan_chunks(
        [
            (3, "MPPA202L Flex Fibula Hook, Standard 1\nMPPA202L Flex Medial Mal Hook 1\nMPPA202L Flex Fibula Hook, Standard 1"),
            (4, "MPPA202L Flex Fibula Hook, Standard 1"),
        ],
        CATALOG,
        ANKLE,
    )

    assert [(row.chunk_index, row.row_text) for row in rows] == [
        (3, "MPPA202L Flex Fibula Hook, Standard 1"),
        (3, "MPPA202L Flex Medial Mal Hook 1"),
        (4, "MPPA202L Flex Fibula Hook, Standard 1"),
    ]


def test_a_document_with_no_family_marks_every_match_out_of_family():
    rows = _rows("MSCF3020 Screw\nMPP202L", family=None)

    assert [(row.sku, row.in_document_family) for row in rows] == [
        ("MSCF3020", False),
        ("MPP202L", None),
    ]


# ---- Storage, against real Postgres ----


class _FixtureGraphClient:
    async def list_catalog_parts(self, skus=None):
        return [
            {"sku": part.sku, "description": part.description, "families": sorted(part.families)}
            for part in _FIXTURE_PARTS
            if skus is None or part.sku in skus
        ]


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * 1536
    vector[index] = 1.0
    return vector


async def _create_document() -> uuid.UUID:
    suffix = uuid.uuid4().hex[:10]
    user = await create_user(f"user-{suffix}@example.com", None, False, True)
    document = await create_document(
        filename=f"doc-{suffix}.pdf", storage_path=f"/tmp/storage-{suffix}.pdf", uploaded_by=user.id
    )
    return document.id


async def _write_chunks(document_id: uuid.UUID, chunks: list[tuple[int, int, str, str]]) -> None:
    async with get_vector_store() as store:
        await store.upsert_chunks(
            document_id,
            [
                {
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "section_title": section_title,
                    "content": content,
                    "embedding": _unit_vector(chunk_index),
                }
                for chunk_index, page_number, section_title, content in chunks
            ],
        )


async def test_index_reads_location_from_chunks_and_description_from_catalog(monkeypatch):
    monkeypatch.setattr(service, "get_graph_client", _FixtureGraphClient)
    document_id = await _create_document()
    try:
        await _write_chunks(
            document_id,
            [
                (0, 2, "Implant options", "Step text with no SKUs."),
                (
                    1,
                    11,
                    "Section 1",
                    (
                        "MIS Bunion Instruments\nItem No. Description Qty.\n"
                        "MSG12150 Ø1.2 x 50mm Guidepin 10\nMPP202L Medium Left 1"
                    ),
                ),
            ],
        )

        await service.index_document_skus(document_id, product_family=MIS)
        skus = {row.sku: row for row in await service.list_document_skus(document_id)}

        assert set(skus) == {"MSG12150", "MPP202L"}
        guidepin = skus["MSG12150"]
        assert (guidepin.chunk_index, guidepin.page_number, guidepin.section_title) == (
            1, 11, "Section 1"
        )
        assert guidepin.citation == f"{document_id}#1"
        assert guidepin.description == "GUIDEPIN, MIS 3.0, 1.2 X 150MM"
        assert guidepin.table_heading == "MIS Bunion Instruments"
        assert guidepin.in_document_family is True
        assert skus["MPP202L"].match_kind == "not_in_catalog"
        assert skus["MPP202L"].description is None
    finally:
        await delete_document(document_id)


async def test_reindex_replaces_rows_and_delete_removes_them(monkeypatch):
    monkeypatch.setattr(service, "get_graph_client", _FixtureGraphClient)
    document_id = await _create_document()
    await _write_chunks(
        document_id, [(0, 1, None, "MSCF3020 Screw"), (1, 2, None, "MSG12150 Guidepin")]
    )
    await service.index_document_skus(document_id, product_family=MIS)
    assert {row.sku for row in await service.list_document_skus(document_id)} == {
        "MSCF3020", "MSG12150"
    }

    await _write_chunks(document_id, [(0, 1, None, "SNAPWIRE Guidewire")])
    await service.index_document_skus(document_id, product_family=MIS)
    assert [(row.sku, row.chunk_index) for row in await service.list_document_skus(document_id)] == [
        ("SNAPWIRE", 0)
    ]

    await delete_document(document_id)
    assert await service.list_document_skus(document_id) == []


class _RecordingLogger:
    """Stands in for the module logger: alembic's fileConfig in conftest's
    session start disables loggers that already exist, so caplog sees
    nothing from this one."""

    def __init__(self):
        self.messages: list[str] = []

    def warning(self, message, *arguments, **keyword_arguments):
        self.messages.append(message % arguments)

    def info(self, message, *arguments, **keyword_arguments):
        self.messages.append(message % arguments)


async def test_a_family_with_no_parts_is_still_indexed_and_logged(monkeypatch):
    monkeypatch.setattr(service, "get_graph_client", _FixtureGraphClient)
    recording_logger = _RecordingLogger()
    monkeypatch.setattr(service, "logger", recording_logger)
    document_id = await _create_document()
    try:
        await _write_chunks(document_id, [(0, 1, None, "MSCF3020 Screw")])

        await service.index_document_skus(document_id, product_family="Unseeded Family")

        assert any(
            message.startswith("document_sku_index_family_has_no_parts")
            for message in recording_logger.messages
        )
        rows = await service.list_document_skus(document_id)
        assert [(row.sku, row.in_document_family) for row in rows] == [("MSCF3020", False)]
    finally:
        await delete_document(document_id)

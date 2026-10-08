"""Exercises backend/agents/tools/document_lookup.py: the name ranking as a
pure function, then the tool itself end to end against real Postgres (same
no-mocking approach as test_vector_store.py)."""
import uuid
from datetime import UTC, datetime

from agents.tools.document_lookup import document_lookup, rank_documents_by_name
from auth.repository import create_user
from documents.repository import DocumentRecord, create_document, delete_document, set_status
from retrieval.vector_store import get_vector_store

_EMBEDDING_DIM = 1536


def _record(filename: str, system_name: str | None = None, document_type_name: str | None = None) -> DocumentRecord:
    now = datetime.now(UTC)
    return DocumentRecord(
        id=uuid.uuid4(),
        filename=filename,
        storage_path=f"/tmp/{filename}",
        status="done",
        error=None,
        uploaded_by=uuid.uuid4(),
        created_at=now,
        updated_at=now,
        system_id=None,
        system_name=system_name,
        document_type_id=None,
        document_type_name=document_type_name,
    )


def _filenames(documents: list[DocumentRecord]) -> list[str]:
    return [document.filename for document in documents]


def test_whole_name_in_order_ranks_above_scattered_word_hits():
    documents = [
        _record("MIS_Control_Inventory_Notes.pdf"),
        _record("MIS-Inventory-Control-Form.pdf"),
        _record("REFLEX Brochure.pdf"),
    ]
    ranked = rank_documents_by_name("the MIS Inventory Control Form", documents)
    assert _filenames(ranked) == ["MIS-Inventory-Control-Form.pdf", "MIS_Control_Inventory_Notes.pdf"]


def test_more_matched_words_rank_above_fewer():
    documents = [_record("MIS Brochure.pdf"), _record("MIS Surgical Technique Guide.pdf")]
    ranked = rank_documents_by_name("MIS surgical technique", documents)
    assert _filenames(ranked) == ["MIS Surgical Technique Guide.pdf", "MIS Brochure.pdf"]


def test_matches_system_and_document_type_tags_not_just_filename():
    documents = [
        _record("Brochure.pdf", system_name="REFLEX - Hybrid", document_type_name="Marketing"),
        _record("Brochure.pdf", system_name="MIS - Foot Recon", document_type_name="Marketing"),
    ]
    ranked = rank_documents_by_name("MIS brochure", documents)
    assert ranked[0].system_name == "MIS - Foot Recon"


def test_filler_words_alone_match_nothing():
    assert rank_documents_by_name("the document for", [_record("The Document.pdf")]) == []


def test_unrelated_documents_are_left_out():
    assert rank_documents_by_name("torque chart", [_record("MIS Brochure.pdf")]) == []


async def test_tool_returns_the_named_document_with_citable_opening_chunks():
    marker = f"zebra{uuid.uuid4().hex[:8]}"
    user = await create_user(f"{marker}@example.com", None, False, True)
    document = await create_document(
        filename=f"{marker}_Inventory_Control_Form.pdf",
        storage_path=f"/tmp/{marker}.pdf",
        uploaded_by=user.id,
    )
    vector = [0.0] * _EMBEDDING_DIM
    vector[40] = 1.0
    try:
        async with get_vector_store() as store:
            await store.upsert_chunks(
                document.id,
                [
                    {"chunk_index": index, "content": f"page {index} text", "embedding": vector}
                    for index in range(3)
                ],
            )

        # Still processing -- not offered until it's actually searchable.
        assert await document_lookup.ainvoke({"name": f"{marker} inventory control form"}) == []

        await set_status(document.id, "done")
        [match] = await document_lookup.ainvoke({"name": f"{marker} inventory control form"})

        assert match["document_id"] == str(document.id)
        assert match["chunk_count"] == 3
        assert [chunk["citation"] for chunk in match["opening_chunks"]] == [
            f"{document.id}#0",
            f"{document.id}#1",
        ]
    finally:
        await delete_document(document.id)

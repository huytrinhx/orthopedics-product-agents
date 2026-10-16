"""Question Types -> Doc Types hierarchy (issue #17): the seed migration
reproduces the old in-code priority mapping, retrieval's filter and rerank
bonus read the table by id, unclassified/unmapped question types retrieve
unfiltered, and the admin routes edit it.
"""
import asyncio
import importlib.util
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

import agents.workflows.deterministic as det
from agents.question_types import QUESTION_TYPE_NAMES
from api.main import app
from config.db import get_connection
from doc_type_hierarchy.repository import RankedDocumentType, list_hierarchy, set_document_types

client = TestClient(app)

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_MIGRATION_REVISION = "b7d3e94a1c58"


def _load_migration():
    [path] = (_BACKEND_ROOT / "migrations" / "versions").glob(f"{_MIGRATION_REVISION}_*.py")
    spec = importlib.util.spec_from_file_location("question_type_document_types_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A frozen copy of the in-code mapping and matching the table replaced
# (deterministic.py before issue #17) -- the seed has to reproduce exactly
# what this did for the question types that existed then.
_LEGACY_PRIORITY: dict[str, list[str]] = {
    "system_contents": ["Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique"],
    "product_characteristics": ["Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique"],
    "compatibility_lookup": ["Inventory Control", "Tray Layout", "Setup Guide", "Surgical Technique"],
    "technique_procedural": ["Surgical Technique", "Setup Guide", "Launch Presentation"],
}
_LEGACY_QUESTION_TYPES = [*_LEGACY_PRIORITY, "documents_lookup"]


def _legacy_bonus(question_types: list[str], document_type_name: str | None) -> float:
    if not document_type_name:
        return 0.0
    for question_type in question_types:
        order = _LEGACY_PRIORITY.get(question_type)
        if not order:
            continue
        for rank, name in enumerate(order):
            if name.lower() in document_type_name.lower():
                return (len(order) - rank) * det._DOCTYPE_BONUS_STEP
        return 0.0
    return 0.0


def _legacy_allowed_ids(question_types: list[str], document_types: list[tuple[str, str]]) -> set[str] | None:
    allowed_names: set[str] = set()
    have_rule = False
    for question_type in question_types:
        order = _LEGACY_PRIORITY.get(question_type)
        if order:
            have_rule = True
            allowed_names.update(name.lower() for name in order)
    if not have_rule:
        return None
    return {
        document_type_id
        for document_type_id, document_type_name in document_types
        if any(name in document_type_name.lower() for name in allowed_names)
    }


# Production-shaped document types: one per legacy name, plus types the
# legacy mapping never mentioned.
_DOCUMENT_TYPES = [
    (str(uuid.uuid4()), name)
    for name in [
        "Inventory Control",
        "Tray Layout",
        "Setup Guide",
        "Surgical Technique",
        "Launch Presentation",
        "Brochure",
        "IFU",
    ]
]
_NAME_BY_ID = dict(_DOCUMENT_TYPES)


def _seeded_ranking() -> dict[str, list[str]]:
    ranking: dict[str, list[str]] = {}
    for question_type, document_type_id, rank in sorted(
        _load_migration().seed_rows(_DOCUMENT_TYPES), key=lambda row: (row[0], row[2])
    ):
        assert rank == len(ranking.get(question_type, []))
        ranking.setdefault(question_type, []).append(document_type_id)
    return ranking


@pytest.mark.parametrize(
    "question_types",
    [[question_type] for question_type in _LEGACY_QUESTION_TYPES]
    + [
        ["technique_procedural", "product_characteristics"],
        ["documents_lookup", "technique_procedural"],
        [],
    ],
)
def test_seeded_mapping_reproduces_the_old_in_code_retrieval_behavior(question_types):
    ranking = _seeded_ranking()

    allowed = det._allowed_document_type_ids(question_types, ranking)
    legacy_allowed = _legacy_allowed_ids(question_types, _DOCUMENT_TYPES)
    assert (set(allowed) if allowed is not None else None) == legacy_allowed

    bonuses = det._document_type_bonuses(question_types, ranking)
    for document_type_id, document_type_name in _DOCUMENT_TYPES:
        assert bonuses.get(document_type_id, 0.0) == pytest.approx(
            _legacy_bonus(question_types, document_type_name)
        )


def test_seed_adds_features_benefits_from_the_stakeholder_csv_and_nothing_for_unfiltered_types():
    ranking = _seeded_ranking()
    assert [_NAME_BY_ID[document_type_id] for document_type_id in ranking["features_benefits"]] == [
        "Brochure",
        "Surgical Technique",
        "Launch Presentation",
    ]
    assert "unclassified" not in ranking
    assert "documents_lookup" not in ranking
    # Every seeded question type is a real one.
    assert set(ranking) <= set(QUESTION_TYPE_NAMES)


async def _create_document_type(name: str) -> str:
    connection = await get_connection()
    try:
        async with connection.cursor() as cursor:
            await cursor.execute("INSERT INTO document_types (name) VALUES (%s) RETURNING id", (name,))
            [document_type_id] = await cursor.fetchone()
        await connection.commit()
        return str(document_type_id)
    finally:
        await connection.close()


async def test_migration_seeds_existing_document_types_into_the_table():
    """Runs the real migration against real document types: down past it,
    create types the seed should pick up, then back up to head."""
    suffix = uuid.uuid4().hex[:8]
    alembic_config = Config(str(_BACKEND_ROOT / "alembic.ini"))
    saved_hierarchy = await list_hierarchy()
    await asyncio.to_thread(command.downgrade, alembic_config, f"{_MIGRATION_REVISION}-1")
    try:
        technique_id = await _create_document_type(f"Surgical Technique {suffix}")
        launch_id = await _create_document_type(f"Launch Presentation {suffix}")
        unrelated_id = await _create_document_type(f"Price List {suffix}")
    finally:
        await asyncio.to_thread(command.upgrade, alembic_config, "head")

    hierarchy = await list_hierarchy()
    technique_ids = [str(document_type.id) for document_type in hierarchy["technique_procedural"]]
    assert technique_ids.index(technique_id) < technique_ids.index(launch_id)
    seeded_ids = {
        str(document_type.id) for document_types in hierarchy.values() for document_type in document_types
    }
    assert unrelated_id not in seeded_ids

    # Put back what other tests had mapped before the round trip.
    for question_type in set(saved_hierarchy) | set(hierarchy):
        await set_document_types(
            question_type,
            [document_type.id for document_type in saved_hierarchy.get(question_type, [])],
        )


class _RecordingVectorSearch:
    def __init__(self, results: list[dict]):
        self.results = results
        self.calls: list[dict] = []

    async def ainvoke(self, arguments: dict) -> list[dict]:
        self.calls.append(arguments)
        return self.results


def _patch_retrieval(monkeypatch, hierarchy: dict[str, list[RankedDocumentType]], results=()):
    async def fake_list_hierarchy():
        return hierarchy

    search = _RecordingVectorSearch(list(results))
    monkeypatch.setattr(det, "list_hierarchy", fake_list_hierarchy)
    monkeypatch.setattr(det, "vector_search", search)
    return search


_TECHNIQUE = RankedDocumentType(uuid.uuid4(), "Surgical Technique")
_SETUP = RankedDocumentType(uuid.uuid4(), "Setup Guide")
_HIERARCHY = {"technique_procedural": [_TECHNIQUE, _SETUP]}


@pytest.mark.parametrize("question_types", [["unclassified"], ["documents_lookup"], []])
async def test_unclassified_and_unmapped_question_types_retrieve_unfiltered(monkeypatch, question_types):
    search = _patch_retrieval(monkeypatch, _HIERARCHY)

    result = await det.hybrid_retrieve({"query": "anything", "resolved_question_type": question_types})

    assert search.calls[0]["document_type_ids"] is None
    assert result["document_type_bonuses"] == {}


async def test_mapped_question_type_filters_by_document_type_id(monkeypatch):
    search = _patch_retrieval(monkeypatch, _HIERARCHY)

    await det.hybrid_retrieve({"query": "steps", "resolved_question_type": ["technique_procedural"]})

    assert search.calls[0]["document_type_ids"] == [str(_TECHNIQUE.id), str(_SETUP.id)]


async def test_rerank_bonus_matches_by_document_type_id_not_name(monkeypatch):
    """A chunk whose doc type *name* still reads "Surgical Technique" but
    whose id isn't mapped gets no bonus -- matching is by id only."""
    unmapped_lookalike_id = str(uuid.uuid4())
    results = [
        {
            "citation": f"doc-{index}#0",
            "document_id": f"doc-{index}",
            "content": "identical text",
            "score": 0.0,
            "document_type": document_type_name,
            "document_type_id": document_type_id,
        }
        for index, (document_type_id, document_type_name) in enumerate(
            [
                (None, None),
                (unmapped_lookalike_id, "Surgical Technique (old)"),
                (str(_SETUP.id), "Setup Guide"),
                (str(_TECHNIQUE.id), "Surgical Technique"),
            ]
        )
    ]
    _patch_retrieval(monkeypatch, _HIERARCHY, results)
    state = {"query": "steps", "resolved_question_type": ["technique_procedural"]}

    state.update(await det.hybrid_retrieve(state))
    reranked = (await det.rerank(state))["reranked"]

    assert [passage["document_type_id"] for passage in reranked[:2]] == [str(_TECHNIQUE.id), str(_SETUP.id)]
    scores = {passage["document_type_id"]: passage["score"] for passage in reranked}
    assert scores[str(_TECHNIQUE.id)] == pytest.approx(2 * det._DOCTYPE_BONUS_STEP)
    assert scores[str(_SETUP.id)] == pytest.approx(det._DOCTYPE_BONUS_STEP)
    assert scores[unmapped_lookalike_id] == 0.0
    assert scores[None] == 0.0


@pytest.mark.parametrize("question_type", ["features_benefits", "unclassified"])
def test_classifier_accepts_the_new_question_types(question_type):
    classification = det._IntentClassification(question_type=[question_type])
    assert classification.question_type == [question_type]


# ---------------- admin routes ----------------


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _token(monkeypatch, tmp_path, *, admin: bool) -> str:
    monkeypatch.setenv("INGEST_DATA_DIR", str(tmp_path))
    email = _unique_email()
    if admin:
        monkeypatch.setenv("ADMIN_EMAILS", email)
    response = client.post("/auth/signup", json={"email": email, "password": "correct horse battery"})
    return response.json()["access_token"]


def _new_document_type(headers: dict, prefix: str) -> dict:
    response = client.post(
        "/document-types", headers=headers, json={"name": f"{prefix}-{uuid.uuid4().hex[:8]}"}
    )
    assert response.status_code == 200
    return response.json()


def _listed(headers: dict, name: str) -> dict:
    listing = client.get("/question-types", headers=headers)
    assert listing.status_code == 200
    return next(question_type for question_type in listing.json() if question_type["name"] == name)


def test_lists_every_code_question_type_with_unclassified_read_only(monkeypatch, tmp_path):
    headers = {"Authorization": f"Bearer {_token(monkeypatch, tmp_path, admin=True)}"}

    listing = client.get("/question-types", headers=headers).json()

    assert [question_type["name"] for question_type in listing] == QUESTION_TYPE_NAMES
    editable = {question_type["name"]: question_type["editable"] for question_type in listing}
    assert editable.pop("unclassified") is False
    assert all(editable.values())
    assert _listed(headers, "unclassified")["document_types"] == []


def test_set_ranks_adds_and_removes_document_types(monkeypatch, tmp_path):
    headers = {"Authorization": f"Bearer {_token(monkeypatch, tmp_path, admin=True)}"}
    original = [document_type["id"] for document_type in _listed(headers, "features_benefits")["document_types"]]
    first = _new_document_type(headers, "Brochure")
    second = _new_document_type(headers, "Launch Presentation")
    try:
        response = client.put(
            "/question-types/features_benefits/document-types",
            headers=headers,
            json={"document_type_ids": [second["id"], first["id"]]},
        )
        assert response.status_code == 200
        assert response.json()["document_types"] == [second, first]

        # Re-rank and drop one.
        client.put(
            "/question-types/features_benefits/document-types",
            headers=headers,
            json={"document_type_ids": [first["id"]]},
        )
        assert _listed(headers, "features_benefits")["document_types"] == [first]
    finally:
        client.put(
            "/question-types/features_benefits/document-types",
            headers=headers,
            json={"document_type_ids": original},
        )


def test_set_rejects_unclassified_unknown_duplicates_and_missing_types(monkeypatch, tmp_path):
    headers = {"Authorization": f"Bearer {_token(monkeypatch, tmp_path, admin=True)}"}
    document_type = _new_document_type(headers, "Brochure")

    def put(name: str, document_type_ids: list[str]):
        return client.put(
            f"/question-types/{name}/document-types",
            headers=headers,
            json={"document_type_ids": document_type_ids},
        )

    assert put("unclassified", [document_type["id"]]).status_code == 400
    assert put("not_a_question_type", [document_type["id"]]).status_code == 404
    assert put("features_benefits", [document_type["id"], document_type["id"]]).status_code == 400
    before = _listed(headers, "features_benefits")["document_types"]
    assert put("features_benefits", [document_type["id"], str(uuid.uuid4())]).status_code == 404
    # A failed replace leaves the list as it was.
    assert _listed(headers, "features_benefits")["document_types"] == before


def test_deleting_a_mapped_document_type_drops_its_mapping(monkeypatch, tmp_path):
    headers = {"Authorization": f"Bearer {_token(monkeypatch, tmp_path, admin=True)}"}
    document_type = _new_document_type(headers, "Brochure")
    question_type = f"test-delete-{uuid.uuid4().hex[:8]}"
    asyncio.run(set_document_types(question_type, [uuid.UUID(document_type["id"])]))

    assert client.delete(f"/document-types/{document_type['id']}", headers=headers).status_code == 204
    assert question_type not in asyncio.run(list_hierarchy())


def test_non_admin_cannot_read_or_edit_the_hierarchy(monkeypatch, tmp_path):
    headers = {"Authorization": f"Bearer {_token(monkeypatch, tmp_path, admin=False)}"}
    assert client.get("/question-types", headers=headers).status_code == 403
    response = client.put(
        "/question-types/features_benefits/document-types",
        headers=headers,
        json={"document_type_ids": []},
    )
    assert response.status_code == 403

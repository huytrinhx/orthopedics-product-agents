"use client";

import { useCallback, useEffect, useState } from "react";
import { listDocumentTypes } from "../../lib/documents/tags/api";
import type { Tag } from "../../lib/documents/tags/types";
import { listQuestionTypes, setQuestionTypeDocumentTypes } from "../../lib/question-types/api";
import type { QuestionType, RankedDocumentType } from "../../lib/question-types/types";

function errorMessage(caughtError: unknown, fallback: string): string {
  return caughtError instanceof Error ? caughtError.message : fallback;
}

function moveItem<Item>(items: Item[], fromIndex: number, toIndex: number): Item[] {
  const reordered = [...items];
  const [moved] = reordered.splice(fromIndex, 1);
  reordered.splice(toIndex, 0, moved);
  return reordered;
}

function QuestionTypeCard({
  questionType,
  documentTypes,
  onSaved,
}: {
  questionType: QuestionType;
  documentTypes: Tag[];
  onSaved: (saved: QuestionType) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [addId, setAddId] = useState("");
  // Removing a doc type stops this question type retrieving from it, so it
  // stops at an in-page confirmation first -- never window.confirm().
  const [confirmingRemoveId, setConfirmingRemoveId] = useState<string | null>(null);

  const ranked = questionType.document_types;
  const addable = documentTypes.filter((candidate) => !ranked.some((listed) => listed.id === candidate.id));
  const editable = questionType.editable && !busy;

  async function save(nextRanked: RankedDocumentType[]) {
    setBusy(true);
    setError(null);
    try {
      onSaved(
        await setQuestionTypeDocumentTypes(
          questionType.name,
          nextRanked.map((documentType) => documentType.id)
        )
      );
    } catch (caughtError) {
      setError(errorMessage(caughtError, "failed to save"));
    } finally {
      setBusy(false);
    }
  }

  function handleMove(fromIndex: number, toIndex: number) {
    if (toIndex < 0 || toIndex >= ranked.length || fromIndex === toIndex) return;
    save(moveItem(ranked, fromIndex, toIndex));
  }

  function handleAdd() {
    const added = documentTypes.find((candidate) => candidate.id === addId);
    if (!added) return;
    setAddId("");
    save([...ranked, { id: added.id, name: added.name }]);
  }

  function handleConfirmRemove(removedId: string) {
    setConfirmingRemoveId(null);
    save(ranked.filter((documentType) => documentType.id !== removedId));
  }

  return (
    <div className="hierarchy-card">
      <div className="hierarchy-card-head">
        <span className="hierarchy-card-name">{questionType.name}</span>
        {!questionType.editable && <span className="badge">Read-only</span>}
        {busy && <span className="workflow-option-saving">Saving…</span>}
      </div>
      <p className="settings-confirm-text">{questionType.description}</p>

      {!questionType.editable ? (
        <p className="settings-confirm-text">
          Always retrieves from every document type, unfiltered — this is how unmapped
          document types stay reachable.
        </p>
      ) : ranked.length === 0 ? (
        <p className="settings-confirm-text">No document types — retrieves unfiltered.</p>
      ) : (
        <ol className="hierarchy-list">
          {ranked.map((documentType, index) => (
            <li
              key={documentType.id}
              className={`hierarchy-item${draggedIndex === index ? " hierarchy-item-dragging" : ""}`}
              draggable={editable}
              onDragStart={(event) => {
                event.dataTransfer.effectAllowed = "move";
                setDraggedIndex(index);
              }}
              onDragOver={(event) => {
                if (draggedIndex !== null) event.preventDefault();
              }}
              onDrop={(event) => {
                event.preventDefault();
                if (draggedIndex !== null) handleMove(draggedIndex, index);
                setDraggedIndex(null);
              }}
              onDragEnd={() => setDraggedIndex(null)}
            >
              <span className="hierarchy-handle" aria-hidden="true">⠿</span>
              <span className="hierarchy-rank">{index + 1}</span>
              <span className="hierarchy-item-name">{documentType.name}</span>
              {confirmingRemoveId === documentType.id ? (
                <span className="actions-cell hierarchy-item-actions">
                  <span className="settings-confirm-text">Stop retrieving from it?</span>
                  <button
                    type="button"
                    className="btn-text settings-danger"
                    disabled={busy}
                    onClick={() => handleConfirmRemove(documentType.id)}
                  >
                    Confirm remove
                  </button>
                  <button type="button" className="btn-text" onClick={() => setConfirmingRemoveId(null)}>
                    Cancel
                  </button>
                </span>
              ) : (
                <span className="actions-cell hierarchy-item-actions">
                  <button
                    type="button"
                    className="btn-text"
                    disabled={!editable || index === 0}
                    aria-label={`Move ${documentType.name} up`}
                    onClick={() => handleMove(index, index - 1)}
                  >
                    ↑
                  </button>
                  <button
                    type="button"
                    className="btn-text"
                    disabled={!editable || index === ranked.length - 1}
                    aria-label={`Move ${documentType.name} down`}
                    onClick={() => handleMove(index, index + 1)}
                  >
                    ↓
                  </button>
                  <button
                    type="button"
                    className="btn-text settings-danger"
                    disabled={!editable}
                    onClick={() => setConfirmingRemoveId(documentType.id)}
                  >
                    Remove
                  </button>
                </span>
              )}
            </li>
          ))}
        </ol>
      )}

      {questionType.editable && (
        <div className="tag-add-row">
          <select
            value={addId}
            onChange={(event) => setAddId(event.target.value)}
            aria-label={`Add a document type to ${questionType.name}`}
            disabled={!editable || addable.length === 0}
          >
            <option value="">{addable.length === 0 ? "Every document type is listed" : "Add document type…"}</option>
            {addable.map((candidate) => (
              <option key={candidate.id} value={candidate.id}>
                {candidate.name}
              </option>
            ))}
          </select>
          <button type="button" className="btn btn-ghost" disabled={!editable || !addId} onClick={handleAdd}>
            Add
          </button>
        </div>
      )}
      {error && <div className="tag-add-error">{error}</div>}
    </div>
  );
}

// Settings section: Question Types -> Doc Types hierarchy (issue #17). Per
// question type, the document types retrieval filters to, most
// authoritative first (rank also sets rerank's bonus). Question types
// themselves are code; only these lists are data.
export function DocTypeHierarchySection() {
  const [questionTypes, setQuestionTypes] = useState<QuestionType[]>([]);
  const [documentTypes, setDocumentTypes] = useState<Tag[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Reloaded on demand too: the Document types section above can rename,
  // merge, or add types without this section knowing.
  const load = useCallback(() => {
    Promise.all([listQuestionTypes(), listDocumentTypes()])
      .then(([fetchedQuestionTypes, fetchedDocumentTypes]) => {
        setQuestionTypes(fetchedQuestionTypes);
        setDocumentTypes(
          [...fetchedDocumentTypes].sort((first, second) => first.name.localeCompare(second.name))
        );
        setLoadError(null);
      })
      .catch((caughtError) => setLoadError(errorMessage(caughtError, "failed to load question types")));
  }, []);

  useEffect(load, [load]);

  function handleSaved(saved: QuestionType) {
    setQuestionTypes((previous) =>
      previous.map((listed) => (listed.name === saved.name ? saved : listed))
    );
  }

  const mappedIds = new Set(
    questionTypes.flatMap((questionType) => questionType.document_types.map((documentType) => documentType.id))
  );

  return (
    <section className="settings-section">
      <h2>Question types → doc types hierarchy</h2>
      <p className="settings-section-lede">
        Which document types each question type retrieves from, most authoritative first —
        drag or use the arrows to rank. A question type with none listed retrieves unfiltered.
        Unmapped document types are only reachable on unclassified questions.{" "}
        <button type="button" className="btn-text" onClick={load}>
          Reload
        </button>
      </p>

      {loadError && <p className="alert" role="alert">{loadError}</p>}

      <div className="hierarchy-cards">
        {questionTypes.map((questionType) => (
          <QuestionTypeCard
            key={questionType.name}
            questionType={questionType}
            documentTypes={documentTypes}
            onSaved={handleSaved}
          />
        ))}
      </div>

      {documentTypes.length > 0 && (
        <>
          <h3 className="hierarchy-coverage-heading">Document type coverage</h3>
          <div className="hierarchy-coverage">
            {documentTypes.map((documentType) => (
              <span key={documentType.id} className="hierarchy-coverage-item">
                {documentType.name}
                {!mappedIds.has(documentType.id) && <span className="badge badge-processing">Unmapped</span>}
              </span>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

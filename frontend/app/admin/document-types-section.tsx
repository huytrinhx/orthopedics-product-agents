"use client";

import { useEffect, useState, type FormEvent } from "react";
import {
  createDocumentType,
  deleteDocumentType,
  listDocumentTypes,
  mergeDocumentType,
  renameDocumentType,
} from "../../lib/documents/tags/api";
import type { Tag } from "../../lib/documents/tags/types";

function sortByName(tags: Tag[]): Tag[] {
  return [...tags].sort((first, second) => first.name.localeCompare(second.name));
}

function errorMessage(caughtError: unknown, fallback: string): string {
  return caughtError instanceof Error ? caughtError.message : fallback;
}

// What a row is currently doing. Merge and delete both stop at an in-page
// confirmation step before anything is sent -- never window.confirm()
// (issue #16).
type RowMode =
  | { kind: "view" }
  | { kind: "renaming" }
  | { kind: "merging" }
  | { kind: "confirmingDelete" };

function DocumentTypeRow({
  documentType,
  otherDocumentTypes,
  onRenamed,
  onMerged,
  onDeleted,
}: {
  documentType: Tag;
  otherDocumentTypes: Tag[];
  onRenamed: (renamed: Tag) => void;
  onMerged: (strayId: string) => void;
  onDeleted: (deletedId: string) => void;
}) {
  const [mode, setMode] = useState<RowMode>({ kind: "view" });
  const [draftName, setDraftName] = useState(documentType.name);
  const [mergeTargetId, setMergeTargetId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mergeTarget = otherDocumentTypes.find((candidate) => candidate.id === mergeTargetId);

  function switchMode(nextMode: RowMode) {
    setMode(nextMode);
    setError(null);
    setDraftName(documentType.name);
    setMergeTargetId("");
  }

  async function handleRename(event: FormEvent) {
    event.preventDefault();
    const name = draftName.trim();
    if (!name || name === documentType.name) {
      switchMode({ kind: "view" });
      return;
    }
    setBusy(true);
    setError(null);
    try {
      onRenamed(await renameDocumentType(documentType.id, name));
      setMode({ kind: "view" });
    } catch (caughtError) {
      setError(errorMessage(caughtError, "failed to rename document type"));
    } finally {
      setBusy(false);
    }
  }

  async function handleConfirmMerge() {
    if (!mergeTarget) return;
    setBusy(true);
    setError(null);
    try {
      await mergeDocumentType(documentType.id, mergeTarget.id);
      onMerged(documentType.id);
    } catch (caughtError) {
      setError(errorMessage(caughtError, "failed to merge document type"));
      setBusy(false);
    }
  }

  async function handleConfirmDelete() {
    setBusy(true);
    setError(null);
    try {
      await deleteDocumentType(documentType.id);
      onDeleted(documentType.id);
    } catch (caughtError) {
      // A type still assigned to a document comes back 409 -- the message
      // says so; merge is the way to retire an in-use type.
      setError(errorMessage(caughtError, "failed to delete document type"));
      setBusy(false);
    }
  }

  return (
    <tr>
      <td>
        {mode.kind === "renaming" ? (
          <form onSubmit={handleRename} className="tag-add-row">
            <input
              value={draftName}
              onChange={(event) => setDraftName(event.target.value)}
              aria-label={`New name for ${documentType.name}`}
              autoFocus
            />
            <button type="submit" className="btn btn-ghost" disabled={busy}>
              {busy ? "Saving…" : "Save"}
            </button>
            <button type="button" className="btn-text" disabled={busy} onClick={() => switchMode({ kind: "view" })}>
              Cancel
            </button>
          </form>
        ) : (
          documentType.name
        )}
        {error && <div className="tag-add-error">{error}</div>}
      </td>
      <td>
        {mode.kind === "view" && (
          <div className="actions-cell">
            <button type="button" className="btn-text" onClick={() => switchMode({ kind: "renaming" })}>
              Rename
            </button>
            <button
              type="button"
              className="btn-text"
              disabled={otherDocumentTypes.length === 0}
              title={otherDocumentTypes.length === 0 ? "No other document type to merge into" : undefined}
              onClick={() => switchMode({ kind: "merging" })}
            >
              Merge…
            </button>
            <button type="button" className="btn-text settings-danger" onClick={() => switchMode({ kind: "confirmingDelete" })}>
              Delete
            </button>
          </div>
        )}

        {mode.kind === "merging" && (
          <div className="settings-confirm">
            <select
              value={mergeTargetId}
              onChange={(event) => setMergeTargetId(event.target.value)}
              aria-label={`Merge ${documentType.name} into`}
              disabled={busy}
            >
              <option value="">Merge into…</option>
              {otherDocumentTypes.map((candidate) => (
                <option key={candidate.id} value={candidate.id}>
                  {candidate.name}
                </option>
              ))}
            </select>
            {mergeTarget && (
              <p className="settings-confirm-text">
                Every document and chunk tagged “{documentType.name}” moves to “{mergeTarget.name}”,
                then “{documentType.name}” is removed. This can’t be undone.
              </p>
            )}
            <div className="actions-cell">
              <button
                type="button"
                className="btn-text settings-danger"
                disabled={!mergeTarget || busy}
                onClick={handleConfirmMerge}
              >
                {busy ? "Merging…" : "Confirm merge"}
              </button>
              <button type="button" className="btn-text" disabled={busy} onClick={() => switchMode({ kind: "view" })}>
                Cancel
              </button>
            </div>
          </div>
        )}

        {mode.kind === "confirmingDelete" && (
          <div className="settings-confirm">
            <p className="settings-confirm-text">Delete “{documentType.name}”? This can’t be undone.</p>
            <div className="actions-cell">
              <button type="button" className="btn-text settings-danger" disabled={busy} onClick={handleConfirmDelete}>
                {busy ? "Deleting…" : "Confirm delete"}
              </button>
              <button type="button" className="btn-text" disabled={busy} onClick={() => switchMode({ kind: "view" })}>
                Cancel
              </button>
            </div>
          </div>
        )}
      </td>
    </tr>
  );
}

// Settings section: Document Types as a closed, admin-curated list (issue
// #16). This is the only place a document type is created, renamed,
// merged, or deleted -- the Document Manager's pickers only choose from it.
export function DocumentTypesSection() {
  const [documentTypes, setDocumentTypes] = useState<Tag[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  useEffect(() => {
    listDocumentTypes()
      .then((fetchedDocumentTypes) => {
        setDocumentTypes(sortByName(fetchedDocumentTypes));
        setLoadError(null);
      })
      .catch((caughtError) => setLoadError(errorMessage(caughtError, "failed to load document types")));
  }, []);

  async function handleCreate(event: FormEvent) {
    event.preventDefault();
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    setCreateError(null);
    try {
      const created = await createDocumentType(name);
      setDocumentTypes((previous) => sortByName([...previous, created]));
      setNewName("");
    } catch (caughtError) {
      setCreateError(errorMessage(caughtError, "failed to create document type"));
    } finally {
      setCreating(false);
    }
  }

  function handleRenamed(renamed: Tag) {
    setDocumentTypes((previous) =>
      sortByName(previous.map((listed) => (listed.id === renamed.id ? renamed : listed)))
    );
  }

  function handleRemoved(removedId: string) {
    setDocumentTypes((previous) => previous.filter((listed) => listed.id !== removedId));
  }

  return (
    <section className="settings-section">
      <h2>Document types</h2>
      <p className="settings-section-lede">
        The list the Document Manager picks from. Merge folds a near-duplicate into the
        canonical type, moving its documents with it. A type still assigned to a document
        can’t be deleted — merge it instead.
      </p>

      {loadError && <p className="alert" role="alert">{loadError}</p>}

      <form onSubmit={handleCreate} className="tag-add-row settings-create-row">
        <input
          value={newName}
          onChange={(event) => setNewName(event.target.value)}
          placeholder="New document type"
          aria-label="New document type"
        />
        <button type="submit" className="btn btn-ghost" disabled={creating || !newName.trim()}>
          {creating ? "Adding…" : "Add"}
        </button>
        {createError && <span className="tag-add-error">{createError}</span>}
      </form>

      {documentTypes.length === 0 ? (
        <div className="empty-state">No document types yet.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {documentTypes.map((documentType) => (
                <DocumentTypeRow
                  key={documentType.id}
                  documentType={documentType}
                  otherDocumentTypes={documentTypes.filter((candidate) => candidate.id !== documentType.id)}
                  onRenamed={handleRenamed}
                  onMerged={handleRemoved}
                  onDeleted={handleRemoved}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

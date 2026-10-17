"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { setDocumentTags } from "../../lib/documents/api";
import type { DocumentRecord } from "../../lib/documents/types";
import type { Tag } from "../../lib/documents/tags/types";
import type { ProductFamily } from "../../lib/catalog/types";

// Per-document tag editor (issue #13), opened from a Documents table row --
// the table itself stays read-only. Family first: the tray list only offers
// that family's trays, and switching family drops any tray that isn't in
// the new one (the server rejects a mismatch either way). Document type is
// pick-from-list only; the list is curated in Settings (issue #16).
// A native <dialog> opened with showModal() gives us the backdrop, Escape
// to close, and focus containment without a modal library.
export function TagModal({
  documentRecord,
  families,
  documentTypes,
  onClose,
  onSaved,
}: {
  documentRecord: DocumentRecord;
  families: ProductFamily[];
  documentTypes: Tag[];
  onClose: () => void;
  onSaved: (updated: DocumentRecord) => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [familyId, setFamilyId] = useState(documentRecord.product_family?.id ?? "");
  const [trayIds, setTrayIds] = useState<string[]>(documentRecord.trays.map((tray) => tray.id));
  const [documentTypeId, setDocumentTypeId] = useState(documentRecord.document_type?.id ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  const selectedFamily = families.find((family) => family.id === familyId) ?? null;

  function handleFamilyChange(nextFamilyId: string) {
    const nextFamily = families.find((family) => family.id === nextFamilyId);
    const nextFamilyTrayIds = new Set(nextFamily?.trays.map((tray) => tray.id) ?? []);
    setFamilyId(nextFamilyId);
    setTrayIds((previous) => previous.filter((trayId) => nextFamilyTrayIds.has(trayId)));
  }

  function toggleTray(trayId: string, checked: boolean) {
    setTrayIds((previous) =>
      checked ? [...previous, trayId] : previous.filter((existingId) => existingId !== trayId)
    );
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const updated = await setDocumentTags(documentRecord.id, {
        documentTypeId: documentTypeId || null,
        family: { productFamilyId: familyId || null, trayIds },
      });
      onSaved(updated);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "failed to save tags");
    } finally {
      setSaving(false);
    }
  }

  return (
    <dialog
      ref={dialogRef}
      className="tag-modal"
      aria-labelledby="tag-modal-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!saving) onClose();
      }}
    >
      <form onSubmit={handleSubmit}>
        <span className="eyebrow">Tags</span>
        <h2 id="tag-modal-title" className="tag-modal-title">
          {documentRecord.filename}
        </h2>

        <label className="field">
          Product family
          <select value={familyId} onChange={(event) => handleFamilyChange(event.target.value)}>
            <option value="">None</option>
            {families.map((family) => (
              <option key={family.id} value={family.id}>
                {family.display_name}
              </option>
            ))}
          </select>
        </label>

        <fieldset className="field tag-modal-trays">
          <legend>Trays</legend>
          {!selectedFamily ? (
            <span className="tag-modal-hint">Choose a product family first.</span>
          ) : selectedFamily.trays.length === 0 ? (
            <span className="tag-modal-hint">This family has no trays.</span>
          ) : (
            <ul className="tag-modal-tray-list">
              {selectedFamily.trays.map((tray) => (
                <li key={tray.id}>
                  <label>
                    <input
                      type="checkbox"
                      checked={trayIds.includes(tray.id)}
                      onChange={(event) => toggleTray(tray.id, event.target.checked)}
                    />
                    {tray.display_name}
                  </label>
                </li>
              ))}
            </ul>
          )}
        </fieldset>

        <label className="field">
          Document type
          <select value={documentTypeId} onChange={(event) => setDocumentTypeId(event.target.value)}>
            <option value="">None</option>
            {documentTypes.map((documentType) => (
              <option key={documentType.id} value={documentType.id}>
                {documentType.name}
              </option>
            ))}
          </select>
        </label>

        {documentRecord.status === "done" && (
          <p className="tag-modal-hint">
            Saving updates this document&apos;s search metadata in place. No reindex needed.
          </p>
        )}
        {error && <p className="alert" role="alert">{error}</p>}

        <div className="tag-modal-actions">
          <button type="button" className="btn btn-ghost" disabled={saving} onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="btn btn-primary" disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </form>
    </dialog>
  );
}

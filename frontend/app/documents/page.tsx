"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import {
  checkSystemHealth,
  deleteDocument,
  DuplicateFilenameError,
  downloadDocumentFile,
  indexDocument,
  listDocuments,
  reuploadDocumentFile,
  uploadDocument,
} from "../../lib/documents/api";
import { listDocumentTypes } from "../../lib/documents/tags/api";
import { listProductFamilies } from "../../lib/catalog/api";
import { useAuth } from "../../lib/auth-context";
import type {
  ComponentHealth,
  DocumentRecord,
  ExistingDocument,
  SystemHealth,
} from "../../lib/documents/types";
import type { Tag } from "../../lib/documents/tags/types";
import type { ProductFamily } from "../../lib/catalog/types";
import { TagModal } from "./tag-modal";

// Polls while anything's still in flight -- the background task (see
// backend/documents/service.py) is a placeholder today (tickets 06/07 wire
// in real ingestion) but real ingestion will take real time, so the list
// needs to reflect status changes without a manual refresh either way.
const POLL_MS = 2000;

function StatusBadge({ documentRecord }: { documentRecord: DocumentRecord }) {
  return (
    <span className={`badge badge-${documentRecord.status}`} title={documentRecord.error ?? undefined}>
      {documentRecord.status}
      {documentRecord.status === "failed" && documentRecord.error ? ` — ${documentRecord.error}` : ""}
    </span>
  );
}

// Re-upload/Download share this outline-icon style (feather-icon shaped,
// no icon library pulled in for two icons) -- the row's action label lives
// in the button's title/aria-label instead of on its face, so these two
// don't force the actions column wider than Index/Reindex and Delete need.
function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}

// One dot per component (backend/api/routes/documents.py's check_system_health
// pokes each for real on every call -- volume file count, a Neo4j query, a
// Postgres/pgvector connection -- rather than reporting a cached status).
// `health` is undefined until the fetch resolves and stays undefined forever
// if it fails outright (see HealthStrip's catch), both shown as "pending".
function HealthItem({ label, health }: { label: string; health?: ComponentHealth }) {
  const state = !health ? "pending" : health.ok ? "ok" : "down";
  const title = health ? health.detail : "Checking…";
  return (
    <span className={`health-item health-${state}`} title={title}>
      <span className="health-dot" />
      {label}
    </span>
  );
}

function HealthStrip({ health, error }: { health: SystemHealth | null; error: string | null }) {
  return (
    <div className="health-row">
      <HealthItem label="Storage" health={health?.volume} />
      <HealthItem label="Graph DB" health={health?.graph_db} />
      <HealthItem label="Vector DB" health={health?.vector_db} />
      {error && <span className="tag-add-error">{error}</span>}
    </div>
  );
}

// Index/Reindex trigger for the (still stubbed, see backend/documents/service.py)
// ingestion pipeline: upload leaves a document "pending" rather than
// auto-indexing it, so this is the only way a document reaches "done".
// Read-only tag cells (issue #12) -- editing family/trays/doc type happens
// in the row's tag modal (tag-modal.tsx). One value per line, wrapped rather than
// truncated, since tray names run long ("Soft Tissue | Comprehensive
// Instrument Tray") and a family can carry several.
function TagValues({ values }: { values: string[] }) {
  if (values.length === 0) return <span className="tag-values-empty">—</span>;
  return (
    <ul className="tag-values">
      {values.map((value, index) => (
        <li key={`${index}-${value}`}>{value}</li>
      ))}
    </ul>
  );
}

// Issue #14: an incomplete document (no Product Family, no Tray, or no
// Document Type) can't be indexed -- the server refuses it too (POST
// /documents/{id}/index answers 409), this just says so up front.
function incompleteExplanation(documentRecord: DocumentRecord): string {
  return `Missing ${documentRecord.missing_tags.join(", ")}. Set them in Tags to index this document.`;
}

function IndexButton({ documentRecord, onIndex }: { documentRecord: DocumentRecord; onIndex: (documentRecord: DocumentRecord) => void }) {
  const running = documentRecord.status === "queued" || documentRecord.status === "processing";
  const incomplete = documentRecord.missing_tags.length > 0;
  const label = running ? "Indexing…" : documentRecord.status === "done" ? "Reindex" : "Index";
  // The title sits on a wrapping span, not the button: a disabled button
  // gets no pointer events in some browsers, so its own tooltip never shows.
  return (
    <span title={incomplete ? incompleteExplanation(documentRecord) : undefined}>
      <button type="button" className="btn-text" disabled={running || incomplete} onClick={() => onIndex(documentRecord)}>
        {label}
      </button>
    </span>
  );
}

// "Last indexed": stamped only by a successful indexing run, so null means
// never indexed -- or indexed before the column existed (no backfill).
function LastIndexed({ documentRecord }: { documentRecord: DocumentRecord }) {
  if (!documentRecord.last_indexed_at) return <span className="tag-values-empty">—</span>;
  return <>{new Date(documentRecord.last_indexed_at).toLocaleString()}</>;
}

// Issue #31: a filename belongs to one document, compared ignoring case
// and surrounding whitespace -- the server's rule (a 409 on upload and
// Re-upload, backed by a unique index). This copy only warns early, against
// the list the page already has; the server's 409 is what enforces it.
function sameFilename(first: string, second: string): boolean {
  return first.trim().toLowerCase() === second.trim().toLowerCase();
}

function asExistingDocument(documentRecord: DocumentRecord): ExistingDocument {
  return {
    id: documentRecord.id,
    filename: documentRecord.filename,
    document_type: documentRecord.document_type,
    product_family: documentRecord.product_family,
    uploaded_at: documentRecord.created_at,
  };
}

// "Surgical Technique · REFLEX, uploaded Oct 9"
function existingDocumentSummary(existing: ExistingDocument): string {
  const tags = [existing.document_type?.name, existing.product_family?.display_name].filter(Boolean);
  const uploaded = new Date(existing.uploaded_at).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return `${tags.length > 0 ? tags.join(" · ") : "untagged"}, uploaded ${uploaded}`;
}

// Stays up until dismissed, listing every refused file -- a multi-file
// upload keeps going past a duplicate, so one alert may name several.
function DuplicateFilenameAlert({
  duplicates,
  onShow,
  onDismiss,
}: {
  duplicates: ExistingDocument[];
  onShow: (documentId: string) => void;
  onDismiss: () => void;
}) {
  return (
    <div className="alert alert-duplicates" role="alert">
      <ul>
        {duplicates.map((existing) => (
          <li key={existing.id}>
            <strong>{existing.filename}</strong> is already uploaded ({existingDocumentSummary(existing)}). To update
            it, use Re-upload on that document.{" "}
            <button type="button" className="btn-text" onClick={() => onShow(existing.id)}>
              Show document
            </button>
          </li>
        ))}
      </ul>
      <button type="button" className="btn-text alert-dismiss" aria-label="Dismiss" onClick={onDismiss}>
        Dismiss
      </button>
    </div>
  );
}

const HIGHLIGHT_MS = 2500;

export default function DocumentsPage() {
  const { user, loading } = useAuth();
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [documentTypes, setDocumentTypes] = useState<Tag[]>([]);
  const [families, setFamilies] = useState<ProductFamily[]>([]);
  const [taggingDocument, setTaggingDocument] = useState<DocumentRecord | null>(null);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Files refused (or skipped) because another document has their name,
  // as that existing document -- see DuplicateFilenameAlert.
  const [duplicates, setDuplicates] = useState<ExistingDocument[]>([]);
  // The early warning for the files currently chosen in the picker.
  const [chosenDuplicates, setChosenDuplicates] = useState<ExistingDocument[]>([]);
  const [highlightedId, setHighlightedId] = useState<string | null>(null);

  // Re-upload's hidden per-row file picker: one shared <input>, retargeted
  // at whichever row's button was clicked (see handleReuploadClick) rather
  // than rendering one file input per row.
  const reuploadInputRef = useRef<HTMLInputElement>(null);
  const reuploadTargetId = useRef<string | null>(null);
  const [reuploadingId, setReuploadingId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
      setListError(null);
    } catch (caughtError) {
      setListError(caughtError instanceof Error ? caughtError.message : "failed to load documents");
    }
  }, []);

  useEffect(() => {
    if (!user?.is_admin) return;
    refresh();
    listDocumentTypes().then(setDocumentTypes).catch(() => {});
    listProductFamilies().then(setFamilies).catch(() => {});
    // A fresh poke every time this page loads (component mount), not a
    // cached/periodic status -- see backend/api/routes/documents.py's
    // check_system_health.
    checkSystemHealth()
      .then((fetchedHealth) => {
        setHealth(fetchedHealth);
        setHealthError(null);
      })
      .catch((caughtError) => {
        setHealth(null);
        setHealthError(caughtError instanceof Error ? caughtError.message : "failed to check system health");
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  useEffect(() => {
    if (!user?.is_admin) return;
    const hasPending = documents.some((listedDocument) => listedDocument.status === "queued" || listedDocument.status === "processing");
    if (!hasPending) return;
    const handle = setInterval(refresh, POLL_MS);
    return () => clearInterval(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, refresh, documents.map((listedDocument) => listedDocument.status).join(",")]);

  useEffect(() => {
    if (!highlightedId) return;
    const handle = setTimeout(() => setHighlightedId(null), HIGHLIGHT_MS);
    return () => clearTimeout(handle);
  }, [highlightedId]);

  function existingDocumentNamed(filename: string, excludingId?: string): DocumentRecord | undefined {
    return documents.find(
      (listedDocument) => listedDocument.id !== excludingId && sameFilename(listedDocument.filename, filename)
    );
  }

  function addDuplicates(found: ExistingDocument[]) {
    if (found.length === 0) return;
    setDuplicates((previous) => [
      ...previous,
      ...found.filter((existing) => !previous.some((shown) => shown.id === existing.id)),
    ]);
  }

  function showDocument(documentId: string) {
    document.getElementById(`document-row-${documentId}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    setHighlightedId(documentId);
  }

  function handleFilesChosen(event: FormEvent<HTMLInputElement>) {
    const chosen = Array.from(event.currentTarget.files ?? []);
    setChosenDuplicates(
      chosen.flatMap((file) => {
        const existing = existingDocumentNamed(file.name);
        return existing ? [asExistingDocument(existing)] : [];
      })
    );
  }

  // Uploads each chosen file in turn. A duplicate doesn't stop the rest:
  // one the page already knows about is skipped without a request, one the
  // server refuses (a document this page hasn't loaded yet) is caught by
  // its 409, and both end up in the duplicates alert.
  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    const files = Array.from(fileInputRef.current?.files ?? []);
    if (files.length === 0) return;
    setUploading(true);
    setUploadError(null);
    const found: ExistingDocument[] = [];
    const failures: string[] = [];
    for (const file of files) {
      const known = existingDocumentNamed(file.name);
      if (known) {
        found.push(asExistingDocument(known));
        continue;
      }
      try {
        await uploadDocument(file);
      } catch (caughtError) {
        if (caughtError instanceof DuplicateFilenameError) {
          found.push(caughtError.existingDocument);
        } else {
          failures.push(`${file.name}: ${caughtError instanceof Error ? caughtError.message : "upload failed"}`);
        }
      }
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
    setChosenDuplicates([]);
    addDuplicates(found);
    if (failures.length > 0) setUploadError(failures.join("; "));
    await refresh();
    setUploading(false);
  }

  async function handleDelete(documentRecord: DocumentRecord) {
    if (!window.confirm(`Delete "${documentRecord.filename}"? This can't be undone.`)) return;
    try {
      await deleteDocument(documentRecord.id);
      setDocuments((previous) => previous.filter((listedDocument) => listedDocument.id !== documentRecord.id));
    } catch (caughtError) {
      setListError(caughtError instanceof Error ? caughtError.message : "failed to delete document");
    }
  }

  async function handleIndex(documentRecord: DocumentRecord) {
    try {
      const updated = await indexDocument(documentRecord.id);
      setDocuments((previous) => previous.map((listedDocument) => (listedDocument.id === documentRecord.id ? updated : listedDocument)));
    } catch (caughtError) {
      setListError(caughtError instanceof Error ? caughtError.message : "failed to trigger indexing");
    }
  }

  function handleTagsSaved(updated: DocumentRecord) {
    setDocuments((previous) => previous.map((listedDocument) => (listedDocument.id === updated.id ? updated : listedDocument)));
    setTaggingDocument(null);
  }

  function handleReuploadClick(documentRecord: DocumentRecord) {
    reuploadTargetId.current = documentRecord.id;
    reuploadInputRef.current?.click();
  }

  async function handleReuploadFileChosen(event: FormEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    const documentId = reuploadTargetId.current;
    event.currentTarget.value = ""; // allow picking the same filename again next time
    if (!file || !documentId) return;
    // Its own name is fine (the usual case); another document's isn't.
    const known = existingDocumentNamed(file.name, documentId);
    if (known) {
      addDuplicates([asExistingDocument(known)]);
      return;
    }
    setReuploadingId(documentId);
    try {
      const updated = await reuploadDocumentFile(documentId, file);
      setDocuments((previous) => previous.map((listedDocument) => (listedDocument.id === documentId ? updated : listedDocument)));
    } catch (caughtError) {
      if (caughtError instanceof DuplicateFilenameError) {
        addDuplicates([caughtError.existingDocument]);
      } else {
        setListError(caughtError instanceof Error ? caughtError.message : "failed to reupload document");
      }
    } finally {
      setReuploadingId(null);
    }
  }

  async function handleDownload(documentRecord: DocumentRecord) {
    try {
      await downloadDocumentFile(documentRecord.id, documentRecord.filename);
    } catch (caughtError) {
      setListError(caughtError instanceof Error ? caughtError.message : "failed to download document");
    }
  }

  if (loading) return <main className="page-wide"><p>Loading…</p></main>;

  if (!user || !user.is_admin) {
    return (
      <main className="page-wide">
        <h1>Document Manager</h1>
        <p>This page is admin-only.</p>
        <Link href="/">Back home</Link>
      </main>
    );
  }

  return (
    <main className="page-wide">
      <span className="eyebrow">Source library</span>
      <h1>Document Manager</h1>
      <p className="lede">
        Upload the technique guides, IFUs, and brochures OrthoMate cites from. A new upload
        lands untagged: open its Tags to set the product family, the trays it covers, and its
        document type (picked from the list curated in <Link href="/admin">Settings</Link>),
        then index it.
      </p>

      <HealthStrip health={health} error={healthError} />

      <form onSubmit={handleUpload} className="upload-row">
        <input ref={fileInputRef} type="file" multiple required onChange={handleFilesChosen} />
        <button type="submit" className="btn btn-primary" disabled={uploading}>
          {uploading ? "Uploading…" : "Upload"}
        </button>
        {chosenDuplicates.length > 0 && (
          <span className="upload-precheck">
            Already uploaded, will be skipped:{" "}
            {chosenDuplicates.map((existing) => existing.filename).join(", ")}
          </span>
        )}
      </form>
      {duplicates.length > 0 && (
        <DuplicateFilenameAlert duplicates={duplicates} onShow={showDocument} onDismiss={() => setDuplicates([])} />
      )}
      {/* Shared hidden picker for every row's Re-upload button -- see
      handleReuploadClick/handleReuploadFileChosen above. */}
      <input
        ref={reuploadInputRef}
        type="file"
        onChange={handleReuploadFileChosen}
        style={{ display: "none" }}
      />
      {uploadError && <p className="alert" role="alert">{uploadError}</p>}
      {listError && <p className="alert" role="alert">{listError}</p>}

      {documents.length === 0 ? (
        <div className="empty-state">No documents uploaded yet.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Filename</th>
                <th>Status</th>
                <th>Product family</th>
                <th>Trays</th>
                <th>Document type</th>
                <th>Uploaded</th>
                <th>Last indexed</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {documents.map((documentRecord) => {
                const incomplete = documentRecord.missing_tags.length > 0;
                return (
                  <tr
                    key={documentRecord.id}
                    id={`document-row-${documentRecord.id}`}
                    className={
                      [
                        incomplete ? "document-row-incomplete" : "",
                        highlightedId === documentRecord.id ? "document-row-highlighted" : "",
                      ]
                        .filter(Boolean)
                        .join(" ") || undefined
                    }
                  >
                    <td>
                      {documentRecord.filename}
                      {incomplete && (
                        <span className="document-incomplete-note">{incompleteExplanation(documentRecord)}</span>
                      )}
                    </td>
                    <td>
                      <StatusBadge documentRecord={documentRecord} />
                    </td>
                    <td className="tag-values-cell">
                      <TagValues values={documentRecord.product_family ? [documentRecord.product_family.display_name] : []} />
                    </td>
                    <td className="tag-values-cell">
                      <TagValues values={documentRecord.trays.map((tray) => tray.display_name)} />
                    </td>
                    <td className="tag-values-cell">
                      <TagValues values={documentRecord.document_type ? [documentRecord.document_type.name] : []} />
                    </td>
                    <td>{new Date(documentRecord.created_at).toLocaleString()}</td>
                    <td>
                      <LastIndexed documentRecord={documentRecord} />
                    </td>
                    <td>
                      <div className="actions-cell">
                        <button type="button" className="btn-text" onClick={() => setTaggingDocument(documentRecord)}>
                          Tags
                        </button>
                        <IndexButton documentRecord={documentRecord} onIndex={handleIndex} />
                        {/* A re-upload re-indexes, so it's behind the same
                        completeness gate as Index/Reindex. */}
                        <span title={incomplete ? incompleteExplanation(documentRecord) : undefined}>
                          <button
                            type="button"
                            className="btn-icon"
                            disabled={incomplete || reuploadingId === documentRecord.id}
                            title={incomplete ? undefined : reuploadingId === documentRecord.id ? "Reuploading…" : "Re-upload"}
                            aria-label={reuploadingId === documentRecord.id ? "Reuploading…" : "Re-upload"}
                            onClick={() => handleReuploadClick(documentRecord)}
                          >
                            <UploadIcon />
                          </button>
                        </span>
                        <button
                          type="button"
                          className="btn-icon"
                          title="Download"
                          aria-label="Download"
                          onClick={() => handleDownload(documentRecord)}
                        >
                          <DownloadIcon />
                        </button>
                        <button
                          type="button"
                          className="btn-icon btn-icon-danger"
                          title="Delete"
                          aria-label="Delete"
                          onClick={() => handleDelete(documentRecord)}
                        >
                          <TrashIcon />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {taggingDocument && (
        <TagModal
          documentRecord={taggingDocument}
          families={families}
          documentTypes={documentTypes}
          onClose={() => setTaggingDocument(null)}
          onSaved={handleTagsSaved}
        />
      )}
    </main>
  );
}

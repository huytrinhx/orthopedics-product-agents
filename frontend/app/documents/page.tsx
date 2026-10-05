"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import {
  checkSystemHealth,
  deleteDocument,
  downloadDocumentFile,
  indexDocument,
  listDocuments,
  reuploadDocumentFile,
  setDocumentTags,
  uploadDocument,
} from "../../lib/documents/api";
import {
  createDocumentType,
  createSystem,
  deleteDocumentType,
  deleteSystem,
  listDocumentTypes,
  listSystems,
} from "../../lib/documents/tags/api";
import { useAuth } from "../../lib/auth-context";
import type { ComponentHealth, DocumentRecord, SystemHealth } from "../../lib/documents/types";
import type { Tag } from "../../lib/documents/tags/types";
import { TagList, TagSelect } from "./tag-select";

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
function IndexButton({ documentRecord, onIndex }: { documentRecord: DocumentRecord; onIndex: (documentRecord: DocumentRecord) => void }) {
  const running = documentRecord.status === "queued" || documentRecord.status === "processing";
  const label = running ? "Indexing…" : documentRecord.status === "done" ? "Reindex" : "Index";
  return (
    <button type="button" className="btn-text" disabled={running} onClick={() => onIndex(documentRecord)}>
      {label}
    </button>
  );
}

export default function DocumentsPage() {
  const { user, loading } = useAuth();
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [systems, setSystems] = useState<Tag[]>([]);
  const [documentTypes, setDocumentTypes] = useState<Tag[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSystemId, setUploadSystemId] = useState("");
  const [uploadDocumentTypeId, setUploadDocumentTypeId] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    listSystems().then(setSystems).catch(() => {});
    listDocumentTypes().then(setDocumentTypes).catch(() => {});
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

  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      await uploadDocument(file, {
        systemId: uploadSystemId || undefined,
        documentTypeId: uploadDocumentTypeId || undefined,
      });
      if (fileInputRef.current) fileInputRef.current.value = "";
      await refresh();
    } catch (caughtError) {
      setUploadError(caughtError instanceof Error ? caughtError.message : "upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleCreateSystem(name: string): Promise<Tag> {
    const tag = await createSystem(name);
    setSystems((previous) => [...previous, tag].sort((first, second) => first.name.localeCompare(second.name)));
    return tag;
  }

  async function handleCreateDocumentType(name: string): Promise<Tag> {
    const tag = await createDocumentType(name);
    setDocumentTypes((previous) => [...previous, tag].sort((first, second) => first.name.localeCompare(second.name)));
    return tag;
  }

  async function handleDeleteSystem(tag: Tag): Promise<void> {
    await deleteSystem(tag.id);
    setSystems((previous) => previous.filter((listedSystem) => listedSystem.id !== tag.id));
  }

  async function handleDeleteDocumentType(tag: Tag): Promise<void> {
    await deleteDocumentType(tag.id);
    setDocumentTypes((previous) => previous.filter((listedDocumentType) => listedDocumentType.id !== tag.id));
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

  function handleReuploadClick(documentRecord: DocumentRecord) {
    reuploadTargetId.current = documentRecord.id;
    reuploadInputRef.current?.click();
  }

  async function handleReuploadFileChosen(event: FormEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    const documentId = reuploadTargetId.current;
    event.currentTarget.value = ""; // allow picking the same filename again next time
    if (!file || !documentId) return;
    setReuploadingId(documentId);
    try {
      const updated = await reuploadDocumentFile(documentId, file);
      setDocuments((previous) => previous.map((listedDocument) => (listedDocument.id === documentId ? updated : listedDocument)));
    } catch (caughtError) {
      setListError(caughtError instanceof Error ? caughtError.message : "failed to reupload document");
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

  async function handleRowTagChange(
    documentRecord: DocumentRecord,
    field: "system" | "document_type",
    tagId: string
  ) {
    const systemId = field === "system" ? tagId || null : documentRecord.system?.id ?? null;
    const documentTypeId = field === "document_type" ? tagId || null : documentRecord.document_type?.id ?? null;
    try {
      const updated = await setDocumentTags(documentRecord.id, { systemId, documentTypeId });
      setDocuments((previous) => previous.map((listedDocument) => (listedDocument.id === documentRecord.id ? updated : listedDocument)));
    } catch (caughtError) {
      setListError(caughtError instanceof Error ? caughtError.message : "failed to update tags");
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
        Upload the technique guides, IFUs, and brochures OrthoMate cites from. Tag each with
        the system and document type it belongs to — reuse an existing tag, or add a new one.
      </p>

      <HealthStrip health={health} error={healthError} />

      {/* TagSelect's own "add new" affordance renders a <form> (see
      tag-select.tsx), so these live outside the upload form rather than
      nested inside it -- nested <form> elements are invalid HTML and
      Next/React will warn (and misbehave) if they're nested here. Selection
      is tracked in state either way, not read from this form's fields. */}
      <div className="tag-picker-row">
        <TagSelect
          label="System"
          tags={systems}
          value={uploadSystemId}
          onChange={setUploadSystemId}
          onCreate={handleCreateSystem}
        />
        <TagSelect
          label="Document type"
          tags={documentTypes}
          value={uploadDocumentTypeId}
          onChange={setUploadDocumentTypeId}
          onCreate={handleCreateDocumentType}
        />
      </div>
      <details className="tag-manager">
        <summary>Manage tags</summary>
        <div className="tag-manager-body">
          <TagList label="System" tags={systems} onDelete={handleDeleteSystem} />
          <TagList label="Document type" tags={documentTypes} onDelete={handleDeleteDocumentType} />
        </div>
      </details>
      <form onSubmit={handleUpload} className="upload-row">
        <input ref={fileInputRef} type="file" required />
        <button type="submit" className="btn btn-primary" disabled={uploading}>
          {uploading ? "Uploading…" : "Upload"}
        </button>
      </form>
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
                <th>System</th>
                <th>Document type</th>
                <th>Uploaded</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {documents.map((documentRecord) => (
                <tr key={documentRecord.id}>
                  <td>{documentRecord.filename}</td>
                  <td>
                    <StatusBadge documentRecord={documentRecord} />
                  </td>
                  <td>
                    <TagSelect
                      label="System"
                      tags={systems}
                      value={documentRecord.system?.id ?? ""}
                      onChange={(id) => handleRowTagChange(documentRecord, "system", id)}
                      onCreate={handleCreateSystem}
                    />
                  </td>
                  <td>
                    <TagSelect
                      label="Document type"
                      tags={documentTypes}
                      value={documentRecord.document_type?.id ?? ""}
                      onChange={(id) => handleRowTagChange(documentRecord, "document_type", id)}
                      onCreate={handleCreateDocumentType}
                    />
                  </td>
                  <td>{new Date(documentRecord.created_at).toLocaleString()}</td>
                  <td>
                    <div className="actions-cell">
                      <IndexButton documentRecord={documentRecord} onIndex={handleIndex} />
                      <button
                        type="button"
                        className="btn-icon"
                        disabled={reuploadingId === documentRecord.id}
                        title={reuploadingId === documentRecord.id ? "Reuploading…" : "Re-upload"}
                        aria-label={reuploadingId === documentRecord.id ? "Reuploading…" : "Re-upload"}
                        onClick={() => handleReuploadClick(documentRecord)}
                      >
                        <UploadIcon />
                      </button>
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
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}

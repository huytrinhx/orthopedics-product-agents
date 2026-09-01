"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import {
  createDocumentType,
  createSystem,
  deleteDocument,
  indexDocument,
  listDocumentTypes,
  listDocuments,
  listSystems,
  setDocumentTags,
  uploadDocument,
} from "../../lib/api";
import { useAuth } from "../../lib/auth-context";
import type { DocumentRecord, Tag } from "../../lib/types";
import { TagSelect } from "./tag-select";

// Polls while anything's still in flight -- the background task (see
// backend/documents/service.py) is a placeholder today (tickets 06/07 wire
// in real ingestion) but real ingestion will take real time, so the list
// needs to reflect status changes without a manual refresh either way.
const POLL_MS = 2000;

function StatusBadge({ doc }: { doc: DocumentRecord }) {
  return (
    <span className={`badge badge-${doc.status}`} title={doc.error ?? undefined}>
      {doc.status}
      {doc.status === "failed" && doc.error ? ` — ${doc.error}` : ""}
    </span>
  );
}

// Index/Reindex trigger for the (still stubbed, see backend/documents/service.py)
// ingestion pipeline: upload leaves a document "pending" rather than
// auto-indexing it, so this is the only way a document reaches "done".
function IndexButton({ doc, onIndex }: { doc: DocumentRecord; onIndex: (doc: DocumentRecord) => void }) {
  const running = doc.status === "queued" || doc.status === "processing";
  const label = running ? "Indexing…" : doc.status === "done" ? "Reindex" : "Index";
  return (
    <button type="button" className="btn-text" disabled={running} onClick={() => onIndex(doc)}>
      {label}
    </button>
  );
}

export default function DocumentsPage() {
  const { user, loading } = useAuth();
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [systems, setSystems] = useState<Tag[]>([]);
  const [documentTypes, setDocumentTypes] = useState<Tag[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSystemId, setUploadSystemId] = useState("");
  const [uploadDocTypeId, setUploadDocTypeId] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
      setListError(null);
    } catch (err) {
      setListError(err instanceof Error ? err.message : "failed to load documents");
    }
  }, []);

  useEffect(() => {
    if (!user?.is_admin) return;
    refresh();
    listSystems().then(setSystems).catch(() => {});
    listDocumentTypes().then(setDocumentTypes).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  useEffect(() => {
    if (!user?.is_admin) return;
    const hasPending = documents.some((d) => d.status === "queued" || d.status === "processing");
    if (!hasPending) return;
    const handle = setInterval(refresh, POLL_MS);
    return () => clearInterval(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, refresh, documents.map((d) => d.status).join(",")]);

  async function handleUpload(e: FormEvent) {
    e.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      await uploadDocument(file, {
        systemId: uploadSystemId || undefined,
        documentTypeId: uploadDocTypeId || undefined,
      });
      if (fileInputRef.current) fileInputRef.current.value = "";
      await refresh();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleCreateSystem(name: string): Promise<Tag> {
    const tag = await createSystem(name);
    setSystems((prev) => [...prev, tag].sort((a, b) => a.name.localeCompare(b.name)));
    return tag;
  }

  async function handleCreateDocumentType(name: string): Promise<Tag> {
    const tag = await createDocumentType(name);
    setDocumentTypes((prev) => [...prev, tag].sort((a, b) => a.name.localeCompare(b.name)));
    return tag;
  }

  async function handleDelete(doc: DocumentRecord) {
    if (!window.confirm(`Delete "${doc.filename}"? This can't be undone.`)) return;
    try {
      await deleteDocument(doc.id);
      setDocuments((prev) => prev.filter((d) => d.id !== doc.id));
    } catch (err) {
      setListError(err instanceof Error ? err.message : "failed to delete document");
    }
  }

  async function handleIndex(doc: DocumentRecord) {
    try {
      const updated = await indexDocument(doc.id);
      setDocuments((prev) => prev.map((d) => (d.id === doc.id ? updated : d)));
    } catch (err) {
      setListError(err instanceof Error ? err.message : "failed to trigger indexing");
    }
  }

  async function handleRowTagChange(
    doc: DocumentRecord,
    field: "system" | "document_type",
    tagId: string
  ) {
    const systemId = field === "system" ? tagId || null : doc.system?.id ?? null;
    const documentTypeId = field === "document_type" ? tagId || null : doc.document_type?.id ?? null;
    try {
      const updated = await setDocumentTags(doc.id, { systemId, documentTypeId });
      setDocuments((prev) => prev.map((d) => (d.id === doc.id ? updated : d)));
    } catch (err) {
      setListError(err instanceof Error ? err.message : "failed to update tags");
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
          value={uploadDocTypeId}
          onChange={setUploadDocTypeId}
          onCreate={handleCreateDocumentType}
        />
      </div>
      <form onSubmit={handleUpload} className="upload-row">
        <input ref={fileInputRef} type="file" required />
        <button type="submit" className="btn btn-primary" disabled={uploading}>
          {uploading ? "Uploading…" : "Upload"}
        </button>
      </form>
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
                <th></th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.id}>
                  <td>{doc.filename}</td>
                  <td>
                    <StatusBadge doc={doc} />
                  </td>
                  <td>
                    <TagSelect
                      label="System"
                      tags={systems}
                      value={doc.system?.id ?? ""}
                      onChange={(id) => handleRowTagChange(doc, "system", id)}
                      onCreate={handleCreateSystem}
                    />
                  </td>
                  <td>
                    <TagSelect
                      label="Document type"
                      tags={documentTypes}
                      value={doc.document_type?.id ?? ""}
                      onChange={(id) => handleRowTagChange(doc, "document_type", id)}
                      onCreate={handleCreateDocumentType}
                    />
                  </td>
                  <td>{new Date(doc.created_at).toLocaleString()}</td>
                  <td>
                    <IndexButton doc={doc} onIndex={handleIndex} />
                  </td>
                  <td>
                    <button
                      type="button"
                      className="btn-text"
                      onClick={() => handleDelete(doc)}
                    >
                      Delete
                    </button>
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

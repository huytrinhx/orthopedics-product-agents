"use client";

// The cited document in a modal, opened from a citation chip: the whole PDF
// at the cited page (pdf-citation-viewer.tsx), or for a non-PDF document
// its chunks with the cited one highlighted. Replaced a side pane beside
// the chat, which squeezed the conversation and redrew the PDF whenever
// the chat column changed width. A native <dialog> opened with
// showModal(), like the Documents page's TagModal, gives the backdrop,
// Escape to close and focus containment without a modal library.
//
// Loading is owned here, not by the chat page: a response for a citation
// that's no longer open is dropped (a fast second click used to show the
// first document's PDF under the second one's title), and the last few
// documents opened stay cached, so going back to one is instant.
import { useEffect, useMemo, useRef, useState } from "react";
import { headingLabel } from "../../lib/chat/format";
import type { ChatCitation } from "../../lib/chat/types";
import { getDocumentChunks, getDocumentFile } from "../../lib/documents/api";
import type { DocumentChunk } from "../../lib/documents/types";
import { PdfCitationViewer } from "./pdf-citation-viewer";

const CACHED_DOCUMENTS = 4;

interface LoadedDocument {
  chunks: DocumentChunk[];
  // The PDF's bytes, or null for a non-PDF document.
  fileBytes: ArrayBuffer | null;
}

// Document id -> its load, most recently opened last. A promise rather than
// the result, so a second click on a document still loading shares it.
const documentCache = new Map<string, Promise<LoadedDocument>>();

function isPdfFilename(filename: string): boolean {
  return filename.toLowerCase().endsWith(".pdf");
}

function loadDocument(documentId: string, isPdf: boolean): Promise<LoadedDocument> {
  const cached = documentCache.get(documentId);
  if (cached) {
    documentCache.delete(documentId);
    documentCache.set(documentId, cached);
    return cached;
  }
  const loading = Promise.all([
    getDocumentChunks(documentId),
    isPdf ? getDocumentFile(documentId).then((blob) => blob.arrayBuffer()) : Promise.resolve(null),
  ]).then(([chunks, fileBytes]) => ({ chunks, fileBytes }));
  documentCache.set(documentId, loading);
  // A failed load isn't cached, so opening the document again retries it.
  loading.catch(() => {
    if (documentCache.get(documentId) === loading) documentCache.delete(documentId);
  });
  while (documentCache.size > CACHED_DOCUMENTS) {
    const oldest = documentCache.keys().next().value;
    if (oldest === undefined) break;
    documentCache.delete(oldest);
  }
  return loading;
}

export function CitationViewerModal({ citation, onClose }: { citation: ChatCitation; onClose: () => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const activeChunkRef = useRef<HTMLDivElement>(null);
  const isPdf = isPdfFilename(citation.filename);
  const [loaded, setLoaded] = useState<{ documentId: string; document: LoadedDocument } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  useEffect(() => {
    let isCurrent = true;
    setError(null);
    loadDocument(citation.document_id, isPdf)
      .then((document) => {
        if (isCurrent) setLoaded({ documentId: citation.document_id, document });
      })
      .catch((caughtError) => {
        if (isCurrent) setError(caughtError instanceof Error ? caughtError.message : "Couldn't load that document");
      });
    return () => {
      isCurrent = false;
    };
  }, [citation.document_id, isPdf]);

  const loadedDocument = loaded?.documentId === citation.document_id ? loaded.document : null;
  const citedChunk = loadedDocument?.chunks.find((chunk) => chunk.chunk_index === citation.chunk_index);
  const citedSection = headingLabel(citation.heading_path, citation.section_title);

  // pdf.js takes ownership of the bytes it's handed (moved to its worker),
  // so it gets its own copy and the cached original stays readable for the
  // next time this document opens. Memoized: a new object would reload it.
  const pdfFile = useMemo(
    () => (loadedDocument?.fileBytes ? { data: new Uint8Array(loadedDocument.fileBytes.slice(0)) } : null),
    [loadedDocument]
  );

  useEffect(() => {
    activeChunkRef.current?.scrollIntoView({ block: "center" });
  }, [loadedDocument]);

  return (
    <dialog
      ref={dialogRef}
      className="citation-modal"
      aria-labelledby="citation-modal-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        // A click on the backdrop lands on the <dialog> itself.
        if (event.target === dialogRef.current) onClose();
      }}
    >
      <div className="citation-modal-header">
        <div className="citation-modal-heading">
          <span id="citation-modal-title" className="citation-modal-title">
            {citation.filename}
          </span>
          {citedSection && <span className="citation-modal-subtitle">{citedSection}</span>}
        </div>
        <button type="button" className="citation-modal-close" onClick={onClose} aria-label="Close document viewer">
          ×
        </button>
      </div>
      {error && (
        <p className="alert" role="alert">
          {error}
        </p>
      )}
      {!error && !loadedDocument && <div className="empty-state">Loading document…</div>}
      {!error && pdfFile && (
        <PdfCitationViewer
          file={pdfFile}
          pageNumber={citedChunk?.page_number ?? 1}
          searchText={citedChunk?.content ?? ""}
          onError={() => setError("This PDF couldn't be displayed.")}
        />
      )}
      {!error && loadedDocument && !isPdf && (
        <div className="citation-modal-chunks">
          {loadedDocument.chunks.map((chunk) => {
            const isActive = chunk.chunk_index === citation.chunk_index;
            return (
              <div
                key={chunk.chunk_index}
                ref={isActive ? activeChunkRef : undefined}
                className={`chat-doc-chunk${isActive ? " chat-doc-chunk-active" : ""}`}
              >
                {headingLabel(chunk.heading_path, chunk.section_title) && (
                  <div className="chat-doc-chunk-section">{headingLabel(chunk.heading_path, chunk.section_title)}</div>
                )}
                <div className="chat-doc-chunk-text">{chunk.content}</div>
              </div>
            );
          })}
        </div>
      )}
    </dialog>
  );
}

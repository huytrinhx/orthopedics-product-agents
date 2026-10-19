import type { Tag } from "./tags/types";

export type DocumentStatus = "pending" | "queued" | "processing" | "done" | "failed";

// Mirrors backend/documents/models.py's DocumentProductFamilyOut and
// catalog/models.py's TrayOut: `name` is the stable catalog name,
// `display_name` the admin-editable label the UI shows.
export interface DocumentProductFamily {
  id: string;
  name: string;
  display_name: string;
}

export interface DocumentTray {
  id: string;
  name: string;
  display_name: string;
}

export interface DocumentRecord {
  id: string;
  filename: string;
  status: DocumentStatus;
  error: string | null;
  uploaded_by: string;
  created_at: string;
  updated_at: string;
  document_type: Tag | null;
  product_family: DocumentProductFamily | null;
  trays: DocumentTray[];
  // Set only when an indexing run finishes successfully (issue #14); null
  // if that's never happened or predates the column.
  last_indexed_at: string | null;
  // Tags still needed before the document may be indexed ("Product
  // Family", "Tray", "Document Type") -- the server's own rule, the same
  // one POST /documents/{id}/index enforces. Empty means complete.
  missing_tags: string[];
}

// Mirrors backend/documents/models.py's ChunkOut -- backs the chat citation
// viewer's right-hand pane (frontend/app/chat/page.tsx).
export interface DocumentChunk {
  chunk_index: number;
  content: string;
  section_title: string | null;
  // The page this chunk's content starts on in the source PDF (ticket 26).
  // null for non-PDF documents and for chunks indexed before this shipped.
  page_number: number | null;
  // The chunk's headings, outermost first (issue #26); section_title is its
  // last element. Empty for chunks indexed before this shipped.
  heading_path: string[];
}

// Mirrors backend/documents/models.py's ComponentHealth/SystemHealthOut.
export interface ComponentHealth {
  ok: boolean;
  detail: string;
}

export interface SystemHealth {
  volume: ComponentHealth;
  graph_db: ComponentHealth;
  vector_db: ComponentHealth;
}

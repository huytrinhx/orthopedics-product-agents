import type { Tag } from "./tags/types";

export type DocumentStatus = "pending" | "queued" | "processing" | "done" | "failed";

export interface DocumentRecord {
  id: string;
  filename: string;
  status: DocumentStatus;
  error: string | null;
  uploaded_by: string;
  created_at: string;
  updated_at: string;
  system: Tag | null;
  document_type: Tag | null;
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

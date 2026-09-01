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
}

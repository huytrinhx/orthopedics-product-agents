export interface AuthUser {
  id: string;
  email: string;
  is_admin: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  user: AuthUser;
}

export type DocumentStatus = "pending" | "queued" | "processing" | "done" | "failed";

export interface Tag {
  id: string;
  name: string;
}

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

export type EvalScores = Partial<{
  faithfulness: number;
  relevance: number;
  style: number;
  citation: number;
}>;

// Mirrors backend/api/routes/chat.py's SSE event payloads.
export type ChatStreamEvent =
  | { event: "thread"; data: { thread_id: string } }
  | { event: "status"; data: { node: string } }
  | { event: "token"; data: { content: string } }
  | {
      event: "done";
      data: {
        thread_id: string;
        answer: string;
        citations: string[];
        eval_scores: EvalScores | null;
      };
    }
  | { event: "error"; data: { message: string } };

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: string[];
  status?: string;
  pending?: boolean;
}

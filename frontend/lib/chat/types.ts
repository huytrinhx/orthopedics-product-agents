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

// Mirrors backend/chat_threads/models.py -- the sidebar (ticket 10).
export interface ChatThread {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatTranscript {
  thread_id: string;
  title: string;
  messages: { role: "user" | "assistant"; content: string }[];
}

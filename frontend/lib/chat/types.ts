export type EvalScores = Partial<{
  faithfulness: number;
  relevance: number;
  style: number;
  citation: number;
}>;

// Mirrors backend/chat_threads/models.py's ChatCitationOut -- a raw
// "{document_id}#{chunk_index}" citation resolved to what the UI needs: a
// filename to show and (document_id, chunk_index) to open the citation
// viewer's right-hand pane on and scroll to.
export interface ChatCitation {
  document_id: string;
  filename: string;
  chunk_index: number;
}

// Mirrors backend/api/routes/chat.py's SSE event payloads.
export type ChatStreamEvent =
  | { event: "thread"; data: { thread_id: string } }
  | { event: "status"; data: { node: string } }
  | { event: "token"; data: { content: string } }
  | {
      event: "clarification";
      data: { thread_id: string; question: string; options: string[] };
    }
  | {
      event: "done";
      data: {
        thread_id: string;
        answer: string;
        citations: ChatCitation[];
        eval_scores: EvalScores | null;
      };
    }
  | { event: "error"; data: { message: string } };

// Ticket 09: detect_intent's interrupt() payload, once resolved to what the
// UI needs to show -- a question plus zero or more clickable options (a
// free-text answer is always accepted too, via the normal chat input).
export interface ChatClarification {
  question: string;
  options: string[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: ChatCitation[];
  status?: string;
  pending?: boolean;
  clarification?: ChatClarification;
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
  messages: { role: "user" | "assistant"; content: string; citations: ChatCitation[] }[];
}

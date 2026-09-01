export type EvalScores = Partial<{
  faithfulness: number;
  relevance: number;
  style: number;
  citation: number;
}>;

// Mirrors backend/feedback/models.py's FeedbackOut -- a human's 4-axis
// score/flag/comment on one specific chat message (ticket 11).
export interface ChatFeedback {
  message_id: string;
  thread_id: string;
  flagged: boolean;
  scores: EvalScores;
  comment: string | null;
  submitted_by: string;
  created_at: string;
  updated_at: string;
}

export interface SubmitFeedbackRequest {
  thread_id: string;
  message_id: string;
  flagged: boolean;
  scores: EvalScores;
  comment?: string | null;
}

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
        // LangGraph's own auto-assigned id for the answer just finalized --
        // what the inline feedback control (ticket 11) keys off of.
        message_id: string;
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
  // Client-generated, stable for the life of the bubble (used as the React
  // key and for progressive updates while a turn is still streaming) --
  // never sent to the backend. Distinct from messageId below.
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: ChatCitation[];
  status?: string;
  pending?: boolean;
  clarification?: ChatClarification;
  // The backend's own id for this message (LangGraph's add_messages uuid) --
  // unset for the optimistic user bubble and for an assistant bubble still
  // streaming; populated once a "done" event lands, or immediately for a
  // message loaded from getChatThread. Ticket 11's feedback control only
  // renders once this is set, and keys its submission off it.
  messageId?: string;
  feedback?: ChatFeedback | null;
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
  messages: {
    message_id: string;
    role: "user" | "assistant";
    content: string;
    citations: ChatCitation[];
    feedback: ChatFeedback | null;
  }[];
}

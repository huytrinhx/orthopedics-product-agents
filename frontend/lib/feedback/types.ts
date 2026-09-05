import type { ChatFeedback } from "../chat/types";

// Mirrors backend/feedback/models.py's FlaggedFeedbackOut -- a flagged
// item's feedback row plus the actual question/answer text, so the Eval
// tab (ticket 15) can list what was flagged without a separate per-row
// transcript fetch.
export interface FlaggedFeedback extends ChatFeedback {
  question: string;
  answer: string;
}

// Mirrors backend/chat_threads/models.py's RerunOut -- one rerun attempt of
// a flagged item. thread_id is a real chat thread: GET /chat/threads/{id}
// and POST /chat/{workflow_name}/resume both work on it unchanged.
export interface Rerun {
  thread_id: string;
  workflow_name: string | null;
  created_at: string;
}

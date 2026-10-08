import type { ChatFeedback } from "../chat/types";

// Mirrors backend/feedback/models.py's FeedbackListItemOut -- one feedback
// row (flagged or not) plus the actual question/answer text and the
// submitter's email, so the Eval tab's table can list everything without a
// separate per-row transcript or user fetch.
export interface FeedbackListItem extends ChatFeedback {
  question: string;
  answer: string;
  submitted_by_email: string | null;
}

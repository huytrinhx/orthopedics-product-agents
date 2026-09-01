"use client";

// Ticket 11: a compact, always-on 4-axis feedback control rendered under
// every finalized assistant message (see chat/page.tsx). One axis per star
// row (0/.25/.5/.75/1, matching backend/agents/judge.py's continuous 0-1
// EvalScores scale more closely than a binary thumbs up/down), a flag
// toggle, and an optional comment -- all sent together on one explicit
// "Save" click (not auto-saved per star, so a half-finished rating never
// fires a request). Keyed by the real backend message_id (chat/types.ts'
// ChatMessage.messageId), not the client-side React key -- resubmitting
// overwrites the previous row (backend/feedback/repository.py's upsert), so
// this never needs its own separate "edit" mode.
import { useState } from "react";
import { submitFeedback } from "../../lib/chat/api";
import type { ChatFeedback, EvalScores } from "../../lib/chat/types";

const AXES: { key: keyof EvalScores; label: string; title: string }[] = [
  { key: "faithfulness", label: "Faithful", title: "Faithfulness -- every claim is supported by the cited sources" },
  { key: "relevance", label: "Relevant", title: "Relevance -- actually addresses the question asked" },
  { key: "style", label: "Style", title: "Style -- clear, concise, well-formatted" },
  { key: "citation", label: "Cited", title: "Citation -- every claim is attributed to a source" },
];

const STAR_VALUES = [0.25, 0.5, 0.75, 1];

function StarRow({
  label,
  title,
  value,
  onChange,
}: {
  label: string;
  title: string;
  value: number | undefined;
  onChange: (value: number) => void;
}) {
  return (
    <div className="feedback-axis" title={title}>
      <span className="feedback-axis-label">{label}</span>
      <span className="feedback-stars">
        {STAR_VALUES.map((starValue) => (
          <button
            key={starValue}
            type="button"
            className="feedback-star"
            aria-label={`${label}: ${starValue * 4} of 4`}
            aria-pressed={value !== undefined && value >= starValue}
            onClick={() => onChange(value === starValue ? 0 : starValue)}
          >
            {value !== undefined && value >= starValue ? "★" : "☆"}
          </button>
        ))}
      </span>
    </div>
  );
}

export function MessageFeedback({
  threadId,
  messageId,
  initialFeedback,
  onSubmitted,
}: {
  threadId: string;
  messageId: string;
  initialFeedback: ChatFeedback | null | undefined;
  onSubmitted: (feedback: ChatFeedback) => void;
}) {
  const [scores, setScores] = useState<EvalScores>(() => initialFeedback?.scores ?? {});
  const [flagged, setFlagged] = useState(() => initialFeedback?.flagged ?? false);
  const [comment, setComment] = useState(() => initialFeedback?.comment ?? "");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [justSaved, setJustSaved] = useState(false);

  const hasContent = flagged || comment.trim().length > 0 || Object.keys(scores).length > 0;

  function setAxis(key: keyof EvalScores, value: number) {
    setJustSaved(false);
    setScores((prev) => {
      if (value === 0) {
        const { [key]: _omit, ...rest } = prev;
        return rest;
      }
      return { ...prev, [key]: value };
    });
  }

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    try {
      const result = await submitFeedback({
        thread_id: threadId,
        message_id: messageId,
        flagged,
        scores,
        comment: comment.trim() || null,
      });
      onSubmitted(result);
      setJustSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't save feedback");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="feedback">
      <div className="feedback-row">
        {AXES.map((axis) => (
          <StarRow
            key={axis.key}
            label={axis.label}
            title={axis.title}
            value={scores[axis.key]}
            onChange={(value) => setAxis(axis.key, value)}
          />
        ))}
        <button
          type="button"
          className={`feedback-flag${flagged ? " feedback-flag-active" : ""}`}
          aria-pressed={flagged}
          title="Flag this answer"
          onClick={() => {
            setJustSaved(false);
            setFlagged((f) => !f);
          }}
        >
          ⚑ Flag
        </button>
      </div>
      <div className="feedback-row">
        <input
          className="feedback-comment"
          value={comment}
          onChange={(e) => {
            setJustSaved(false);
            setComment(e.target.value);
          }}
          placeholder="Optional comment…"
        />
        <button
          type="button"
          className="btn btn-ghost feedback-submit"
          onClick={handleSubmit}
          disabled={submitting || !hasContent}
        >
          {submitting ? "Saving…" : justSaved ? "Saved" : "Save feedback"}
        </button>
      </div>
      {error && <div className="feedback-error">{error}</div>}
    </div>
  );
}

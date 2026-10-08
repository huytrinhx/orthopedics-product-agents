"use client";

// One feedback row in the Eval tab's table (ticket 15): when, who, the
// question, the four human scores, the comment, flagged/resolved, and
// Rerun/Delete. Clicking the row expands a second row underneath with the
// full question and answer. Resolved is only offered on a flagged row --
// it means "the flagged issue was confirmed fixed" (the backend 404s it on
// anything else).
import { Fragment, useState } from "react";
import { deleteFeedback, setFeedbackResolved } from "../../lib/feedback/api";
import type { FeedbackListItem } from "../../lib/feedback/types";

const SCORE_AXES = ["faithfulness", "relevance", "style", "citation"] as const;

export const COLUMN_COUNT = 11;

function formatScore(value: number | undefined): string {
  return value === undefined ? "—" : value.toFixed(2);
}

export function FeedbackRow({
  item,
  onRerun,
  onResolvedChanged,
  onDeleted,
}: {
  item: FeedbackListItem;
  onRerun: (question: string) => void;
  onResolvedChanged: (resolved: boolean) => void;
  onDeleted: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  async function toggleResolved() {
    setResolving(true);
    setActionError(null);
    try {
      const next = !item.resolved;
      await setFeedbackResolved(item.message_id, next);
      onResolvedChanged(next);
    } catch (caughtError) {
      setActionError(caughtError instanceof Error ? caughtError.message : "Couldn't update this item");
    } finally {
      setResolving(false);
    }
  }

  async function handleDelete() {
    setDeleting(true);
    setActionError(null);
    try {
      await deleteFeedback(item.message_id);
      onDeleted();
    } catch (caughtError) {
      setActionError(caughtError instanceof Error ? caughtError.message : "Couldn't delete this item");
      setDeleting(false);
    }
  }

  return (
    <Fragment>
      <tr
        className={`eval-row${expanded ? " eval-row-expanded" : ""}${item.resolved ? " eval-row-resolved" : ""}`}
        onClick={() => setExpanded((previous) => !previous)}
      >
        <td className="eval-cell-nowrap">{new Date(item.created_at).toLocaleString()}</td>
        <td>{item.submitted_by_email ?? "—"}</td>
        <td className="eval-cell-question" title={item.question}>
          {item.question}
        </td>
        {SCORE_AXES.map((axis) => (
          <td key={axis} className="eval-cell-score">
            {formatScore(item.scores[axis])}
          </td>
        ))}
        <td className="eval-cell-comment">{item.comment ?? "—"}</td>
        <td>{item.flagged ? <span className="badge badge-failed">Flagged</span> : "—"}</td>
        <td onClick={(event) => event.stopPropagation()}>
          {item.flagged ? (
            <input
              type="checkbox"
              aria-label="Resolved"
              checked={item.resolved}
              disabled={resolving}
              onChange={toggleResolved}
            />
          ) : (
            "—"
          )}
        </td>
        <td className="eval-cell-actions" onClick={(event) => event.stopPropagation()}>
          <button type="button" className="btn-text" onClick={() => onRerun(item.question)}>
            Rerun
          </button>
          {confirmingDelete ? (
            <span className="eval-delete-confirm">
              <button type="button" className="btn-text eval-delete" onClick={handleDelete} disabled={deleting}>
                {deleting ? "Deleting…" : "Confirm"}
              </button>
              <button type="button" className="btn-text" onClick={() => setConfirmingDelete(false)} disabled={deleting}>
                Cancel
              </button>
            </span>
          ) : (
            <button type="button" className="btn-text eval-delete" onClick={() => setConfirmingDelete(true)}>
              Delete
            </button>
          )}
        </td>
      </tr>
      {actionError && (
        <tr>
          <td colSpan={COLUMN_COUNT}>
            <p className="alert" role="alert">{actionError}</p>
          </td>
        </tr>
      )}
      {expanded && (
        <tr className="eval-detail">
          <td colSpan={COLUMN_COUNT}>
            <p className="eval-detail-label">Question</p>
            <p className="eval-detail-text">{item.question}</p>
            <p className="eval-detail-label">Answer</p>
            <p className="eval-detail-text">{item.answer}</p>
            {item.comment && (
              <>
                <p className="eval-detail-label">Comment</p>
                <p className="eval-detail-text">{item.comment}</p>
              </>
            )}
          </td>
        </tr>
      )}
    </Fragment>
  );
}

"use client";

// One flagged feedback item on the Eval tab (ticket 15): the question/
// answer that was flagged, its human scores, a resolved toggle, a workflow
// picker + "Rerun" trigger, and that item's own rerun history -- each
// rerun expandable inline via RerunConversation, reusing the real chat UI.
import { useState } from "react";
import type { WorkflowInfo } from "../../lib/admin/types";
import { deleteFlaggedFeedback, listReruns, setFeedbackResolved } from "../../lib/feedback/api";
import type { FlaggedFeedback, Rerun } from "../../lib/feedback/types";
import { RerunConversation } from "./rerun-conversation";

function formatScore(value: number | undefined): string {
  return value === undefined ? "—" : value.toFixed(2);
}

export function FlaggedItem({
  item,
  workflows,
  onDeleted,
}: {
  item: FlaggedFeedback;
  workflows: WorkflowInfo[];
  onDeleted: () => void;
}) {
  const [resolved, setResolved] = useState(item.resolved);
  const [resolving, setResolving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [reruns, setReruns] = useState<Rerun[] | null>(null);
  const [rerunsError, setRerunsError] = useState<string | null>(null);
  const [expandedThreadId, setExpandedThreadId] = useState<string | null>(null);
  const [workflowName, setWorkflowName] = useState(workflows[0]?.name ?? "");
  // Set once "Rerun" is clicked, filled in once the new thread's id arrives
  // -- kept separate from `reruns` (the fetched-history list) so the live
  // attempt is never rendered twice once it also shows up in a later
  // history fetch.
  const [liveRerun, setLiveRerun] = useState<{ threadId: string | null; workflowName: string } | null>(
    null
  );

  async function loadReruns() {
    try {
      setReruns(await listReruns(item.message_id));
    } catch (err) {
      setRerunsError(err instanceof Error ? err.message : "Couldn't load rerun history");
    }
  }

  async function toggleResolved() {
    setResolving(true);
    try {
      const next = !resolved;
      await setFeedbackResolved(item.message_id, next);
      setResolved(next);
    } catch {
      // Leave the checkbox at its prior value -- the state above was never
      // optimistically flipped, so nothing to roll back.
    } finally {
      setResolving(false);
    }
  }

  const historyRows = (reruns ?? []).filter((r) => r.thread_id !== liveRerun?.threadId);

  async function handleDelete() {
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteFlaggedFeedback(item.message_id);
      onDeleted();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Couldn't delete this item");
      setDeleting(false);
    }
  }

  return (
    <div className={`flagged-item${resolved ? " flagged-item-resolved" : ""}`}>
      <div className="flagged-item-head">
        <div className="flagged-item-qa">
          <p className="flagged-item-question">{item.question}</p>
          <p className="flagged-item-answer">{item.answer}</p>
        </div>
        <div className="flagged-item-head-actions">
          <label className="flagged-resolved-toggle">
            <input type="checkbox" checked={resolved} disabled={resolving} onChange={toggleResolved} />
            Resolved
          </label>
          {confirmingDelete ? (
            <span className="flagged-delete-confirm">
              Delete this item?
              <button type="button" className="btn-text" onClick={handleDelete} disabled={deleting}>
                {deleting ? "Deleting…" : "Yes, delete"}
              </button>
              <button type="button" className="btn-text" onClick={() => setConfirmingDelete(false)} disabled={deleting}>
                Cancel
              </button>
            </span>
          ) : (
            <button type="button" className="btn-text flagged-delete" onClick={() => setConfirmingDelete(true)}>
              Delete
            </button>
          )}
        </div>
      </div>
      {deleteError && <p className="alert" role="alert">{deleteError}</p>}

      <div className="flagged-item-meta">
        <span>Faithfulness {formatScore(item.scores.faithfulness)}</span>
        <span>Relevance {formatScore(item.scores.relevance)}</span>
        <span>Style {formatScore(item.scores.style)}</span>
        <span>Citation {formatScore(item.scores.citation)}</span>
        {item.comment && <span className="flagged-item-comment">&ldquo;{item.comment}&rdquo;</span>}
      </div>

      <div className="flagged-item-actions">
        <select
          className="flagged-workflow-select"
          value={workflowName}
          onChange={(e) => setWorkflowName(e.target.value)}
          disabled={liveRerun !== null}
        >
          {workflows.map((w) => (
            <option key={w.name} value={w.name}>
              {w.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn btn-ghost"
          onClick={() => setLiveRerun({ threadId: null, workflowName })}
          disabled={liveRerun !== null || !workflowName}
        >
          Rerun
        </button>
        <button
          type="button"
          className="btn-text"
          onClick={() => {
            if (reruns === null) loadReruns();
          }}
        >
          {reruns === null
            ? "Show rerun history"
            : `${historyRows.length} past rerun${historyRows.length === 1 ? "" : "s"}`}
        </button>
      </div>

      {liveRerun && (
        <div className="rerun-attempt">
          <div className="rerun-attempt-label">Rerun just now &middot; {liveRerun.workflowName}</div>
          <RerunConversation
            mode="live"
            originalThreadId={item.thread_id}
            originalMessageId={item.message_id}
            workflowName={liveRerun.workflowName}
            onThreadCreated={(threadId) => {
              setLiveRerun((prev) => (prev ? { ...prev, threadId } : prev));
              setReruns((prev) => [
                { thread_id: threadId, workflow_name: liveRerun.workflowName, created_at: new Date().toISOString() },
                ...(prev ?? []),
              ]);
            }}
          />
        </div>
      )}

      {rerunsError && <p className="alert" role="alert">{rerunsError}</p>}
      {historyRows.length > 0 && (
        <div className="rerun-history">
          {historyRows.map((r) => (
            <div key={r.thread_id} className="rerun-attempt">
              <button
                type="button"
                className="btn-text rerun-attempt-label"
                onClick={() => setExpandedThreadId((prev) => (prev === r.thread_id ? null : r.thread_id))}
              >
                {new Date(r.created_at).toLocaleString()} &middot; {r.workflow_name ?? "unknown workflow"}
              </button>
              {expandedThreadId === r.thread_id && (
                <RerunConversation mode="history" threadId={r.thread_id} workflowName={r.workflow_name} />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

"use client";

// Ticket 15 (Eval tab): every piece of feedback reps have given -- scores,
// comments, and flags -- in one table, newest first. "Unresolved" (the
// default) narrows it to flagged items nobody has confirmed fixed yet;
// "All" shows everything. Rerun opens the chat tab with the rated question
// sent fresh in a new conversation (the `?ask=` handling in
// app/chat/page.tsx), on whatever workflow is the current default -- so
// the admin sees exactly what a rep would get now. This replaced an inline,
// history-replaying rerun with a per-item attempt history (2026-10-07, see
// build-log.md).
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useAuth } from "../../lib/auth-context";
import { listFeedback } from "../../lib/feedback/api";
import type { FeedbackListItem } from "../../lib/feedback/types";
import { FeedbackRow } from "./feedback-row";

type Filter = "unresolved" | "all";

function isUnresolved(item: FeedbackListItem): boolean {
  return item.flagged && !item.resolved;
}

export default function EvalsPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [feedback, setFeedback] = useState<FeedbackListItem[] | null>(null);
  const [filter, setFilter] = useState<Filter>("unresolved");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user?.is_admin) return;
    listFeedback()
      .then(setFeedback)
      .catch((caughtError) => setError(caughtError instanceof Error ? caughtError.message : "Couldn't load feedback"));
  }, [user]);

  function updateItem(messageId: string, patch: Partial<FeedbackListItem>) {
    setFeedback((previous) =>
      previous ? previous.map((item) => (item.message_id === messageId ? { ...item, ...patch } : item)) : previous
    );
  }

  function removeItem(messageId: string) {
    setFeedback((previous) => (previous ? previous.filter((item) => item.message_id !== messageId) : previous));
  }

  function rerun(question: string) {
    router.push(`/chat?ask=${encodeURIComponent(question)}`);
  }

  if (loading) return <main className="page-wide"><p>Loading…</p></main>;

  if (!user || !user.is_admin) {
    return (
      <main className="page-wide">
        <h1>Eval tab</h1>
        <p>This page is admin-only.</p>
        <Link href="/">Back home</Link>
      </main>
    );
  }

  const visible = feedback === null ? [] : filter === "all" ? feedback : feedback.filter(isUnresolved);
  const unresolvedCount = feedback === null ? 0 : feedback.filter(isUnresolved).length;

  return (
    <main className="page-wide">
      <span className="eyebrow">Feedback review</span>
      <h1>Eval tab</h1>
      <p className="lede">
        Every score, comment, and flag reps have left on an answer, newest first. Click a row to read
        the full answer. Rerun asks the same question again in a fresh chat on the current default
        workflow; mark a flagged item resolved once the rerun confirms it&rsquo;s fixed.
      </p>

      <div className="eval-filter" role="group" aria-label="Filter feedback">
        <button
          type="button"
          className={`btn ${filter === "unresolved" ? "btn-primary" : "btn-ghost"}`}
          aria-pressed={filter === "unresolved"}
          onClick={() => setFilter("unresolved")}
        >
          Unresolved ({unresolvedCount})
        </button>
        <button
          type="button"
          className={`btn ${filter === "all" ? "btn-primary" : "btn-ghost"}`}
          aria-pressed={filter === "all"}
          onClick={() => setFilter("all")}
        >
          All ({feedback?.length ?? 0})
        </button>
      </div>

      {error && <p className="alert" role="alert">{error}</p>}

      {feedback === null && !error && <p>Loading feedback…</p>}
      {feedback !== null && visible.length === 0 && (
        <p className="empty-state">
          {filter === "unresolved" ? "No unresolved flagged feedback." : "No feedback yet."}
        </p>
      )}
      {visible.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Created</th>
                <th>Submitted by</th>
                <th>Question</th>
                <th>Faithfulness</th>
                <th>Relevance</th>
                <th>Style</th>
                <th>Citation</th>
                <th>Comment</th>
                <th>Flagged</th>
                <th>Resolved</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {visible.map((item) => (
                <FeedbackRow
                  key={item.message_id}
                  item={item}
                  onRerun={rerun}
                  onResolvedChanged={(resolved) => updateItem(item.message_id, { resolved })}
                  onDeleted={() => removeItem(item.message_id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}

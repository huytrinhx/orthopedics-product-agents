"use client";

// Ticket 15 (Eval tab) -- rescoped 2026-09-05 from an eval-dataset harness
// into a feedback-rerun tool after a grilling session found the original
// idea's premise (a static golden-dataset CI harness) undermined by every
// workflow's own clarification-pause behavior. What's here instead: every
// flagged feedback item, an admin can re-ask the flagged question against
// the current code (in its real conversational context, against a chosen
// workflow) and see the new answer next to the old one, and mark an item
// resolved once a rerun confirms it's fixed.
import Link from "next/link";
import { useEffect, useState } from "react";
import { getAdminSettings } from "../../lib/admin/api";
import type { WorkflowInfo } from "../../lib/admin/types";
import { useAuth } from "../../lib/auth-context";
import { listFlaggedFeedback } from "../../lib/feedback/api";
import type { FlaggedFeedback } from "../../lib/feedback/types";
import { FlaggedItem } from "./flagged-item";

export default function EvalsPage() {
  const { user, loading } = useAuth();
  const [flagged, setFlagged] = useState<FlaggedFeedback[] | null>(null);
  const [workflows, setWorkflows] = useState<WorkflowInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user?.is_admin) return;
    listFlaggedFeedback()
      .then(setFlagged)
      .catch((err) => setError(err instanceof Error ? err.message : "Couldn't load flagged feedback"));
    getAdminSettings()
      .then((s) => setWorkflows(s.workflows.filter((w) => w.functional)))
      .catch(() => {});
  }, [user]);

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

  return (
    <main className="page-wide">
      <span className="eyebrow">Feedback review</span>
      <h1>Eval tab</h1>
      <p className="lede">
        Every flagged answer, newest first. Rerun a flagged question against the current code to check
        whether a fix actually worked -- each rerun replays the real conversation and keeps a history of
        attempts.
      </p>

      {error && <p className="alert" role="alert">{error}</p>}

      {flagged === null && !error && <p>Loading flagged feedback…</p>}
      {flagged !== null && flagged.length === 0 && (
        <p className="empty-state">No flagged feedback yet.</p>
      )}
      {flagged !== null && flagged.length > 0 && (
        <div className="flagged-list">
          {flagged.map((item) => (
            <FlaggedItem key={item.message_id} item={item} workflows={workflows} />
          ))}
        </div>
      )}
    </main>
  );
}

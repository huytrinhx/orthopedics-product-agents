"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { getAdminSettings, setDefaultWorkflow } from "../../lib/admin/api";
import { useAuth } from "../../lib/auth-context";
import type { AdminSettings } from "../../lib/admin/types";

export default function AdminPage() {
  const { user, loading } = useAuth();
  const [settings, setSettings] = useState<AdminSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Tracks which workflow a PUT is in flight for, so only that radio (not
  // the whole picker) disables while saving.
  const [saving, setSaving] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setSettings(await getAdminSettings());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to load settings");
    }
  }, []);

  useEffect(() => {
    if (!user?.is_admin) return;
    refresh();
  }, [user, refresh]);

  // Saves as soon as a workflow is picked -- there's only this one field,
  // so a separate Save button would just be extra state to manage for no
  // benefit.
  async function handleSelect(workflowName: string) {
    if (!settings || workflowName === settings.default_workflow) return;
    setSaving(workflowName);
    try {
      setSettings(await setDefaultWorkflow(workflowName));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to update default workflow");
    } finally {
      setSaving(null);
    }
  }

  if (loading) return <main className="page-wide"><p>Loading…</p></main>;

  if (!user || !user.is_admin) {
    return (
      <main className="page-wide">
        <h1>Settings</h1>
        <p>This page is admin-only.</p>
        <Link href="/">Back home</Link>
      </main>
    );
  }

  return (
    <main className="page-wide">
      <span className="eyebrow">Admin</span>
      <h1>Settings</h1>
      <p className="lede">
        Choose which chat workflow new conversations use. This is the only place workflow
        choice is ever exposed — chat users never see or pick it themselves.
      </p>

      {error && <p className="alert" role="alert">{error}</p>}

      {settings && (
        <div className="workflow-picker">
          {settings.workflows.map((wf) => (
            <label
              key={wf.name}
              className={`workflow-option${wf.functional ? "" : " workflow-option-disabled"}`}
            >
              <input
                type="radio"
                name="default_workflow"
                value={wf.name}
                checked={settings.default_workflow === wf.name}
                disabled={!wf.functional || saving !== null}
                onChange={() => handleSelect(wf.name)}
              />
              <span className="workflow-option-name">{wf.name}</span>
              {!wf.functional && <span className="badge">Coming soon</span>}
              {saving === wf.name && <span className="workflow-option-saving">Saving…</span>}
            </label>
          ))}
        </div>
      )}
    </main>
  );
}

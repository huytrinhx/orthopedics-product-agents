"use client";

import { useEffect, useState } from "react";
import { getAdminSettings, setDefaultWorkflow } from "../../lib/admin/api";
import type { AdminSettings } from "../../lib/admin/types";

// Settings section: which chat workflow new conversations default to
// (ticket 14). The only place workflow choice is ever exposed.
export function WorkflowSection() {
  const [settings, setSettings] = useState<AdminSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Tracks which workflow a PUT is in flight for, so only that radio (not
  // the whole picker) disables while saving.
  const [saving, setSaving] = useState<string | null>(null);

  useEffect(() => {
    getAdminSettings()
      .then((fetchedSettings) => {
        setSettings(fetchedSettings);
        setError(null);
      })
      .catch((caughtError) => {
        setError(caughtError instanceof Error ? caughtError.message : "failed to load settings");
      });
  }, []);

  // Saves as soon as a workflow is picked -- there's only this one field,
  // so a separate Save button would just be extra state to manage for no
  // benefit.
  async function handleSelect(workflowName: string) {
    if (!settings || workflowName === settings.default_workflow) return;
    setSaving(workflowName);
    try {
      setSettings(await setDefaultWorkflow(workflowName));
      setError(null);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "failed to update default workflow");
    } finally {
      setSaving(null);
    }
  }

  return (
    <section className="settings-section">
      <h2>Chat workflow</h2>
      <p className="settings-section-lede">
        Choose which chat workflow new conversations use. This is the only place workflow
        choice is ever exposed — chat users never see or pick it themselves.
      </p>

      {error && <p className="alert" role="alert">{error}</p>}

      {settings && (
        <div className="workflow-picker">
          {settings.workflows.map((workflow) => (
            <label
              key={workflow.name}
              className={`workflow-option${workflow.functional ? "" : " workflow-option-disabled"}`}
            >
              <input
                type="radio"
                name="default_workflow"
                value={workflow.name}
                checked={settings.default_workflow === workflow.name}
                disabled={!workflow.functional || saving !== null}
                onChange={() => handleSelect(workflow.name)}
              />
              <span className="workflow-option-name">{workflow.name}</span>
              {!workflow.functional && <span className="badge">Coming soon</span>}
              {saving === workflow.name && <span className="workflow-option-saving">Saving…</span>}
            </label>
          ))}
        </div>
      )}
    </section>
  );
}

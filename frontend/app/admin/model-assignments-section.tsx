"use client";

import { useEffect, useState } from "react";
import {
  getModelAssignments,
  resetModelAssignment,
  saveModelAssignment,
  testModelAssignment,
} from "../../lib/model-assignments/api";
import type {
  ModelAssignments,
  ModelChoice,
  ModelSlot,
  ProbeResult,
} from "../../lib/model-assignments/types";
import { ModelPicker } from "./model-picker";

function errorMessage(caughtError: unknown, fallback: string): string {
  return caughtError instanceof Error ? caughtError.message : fallback;
}

const NEED_LABELS: Record<string, string> = {
  structured_output: "Structured output",
  streamed_text: "Streamed text",
  tool_calling: "Tool calling",
  image_input: "Image input",
};

function describeChoice(choice: ModelChoice): string {
  return `${choice.model_id} · reasoning effort: ${choice.reasoning_effort ?? "model default"}`;
}

// One slot: what it runs now, who changed it, and (unless read-only) an
// editor whose Save runs the slot's live probe first.
function SlotCard({
  slot,
  defaultModel,
  readOnly,
  onSaved,
}: {
  slot: ModelSlot;
  defaultModel: string;
  readOnly: boolean;
  onSaved: (assignments: ModelAssignments) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<ModelChoice | null>(null);
  const [busy, setBusy] = useState<"saving" | "testing" | "resetting" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [probeResult, setProbeResult] = useState<ProbeResult | null>(null);

  function startEditing() {
    setDraft(
      slot.assignment
        ? { model_id: slot.assignment.model_id, reasoning_effort: slot.assignment.reasoning_effort }
        : null
    );
    setError(null);
    setProbeResult(null);
    setEditing(true);
  }

  async function save() {
    if (!draft) return;
    setBusy("saving");
    setError(null);
    setProbeResult(null);
    try {
      onSaved(await saveModelAssignment(slot.slot, draft));
      setEditing(false);
    } catch (caughtError) {
      setError(errorMessage(caughtError, "failed to save the model"));
    } finally {
      setBusy(null);
    }
  }

  async function test() {
    if (!draft) return;
    setBusy("testing");
    setError(null);
    setProbeResult(null);
    try {
      setProbeResult(await testModelAssignment(slot.slot, draft));
    } catch (caughtError) {
      setError(errorMessage(caughtError, "failed to test the model"));
    } finally {
      setBusy(null);
    }
  }

  async function reset() {
    setBusy("resetting");
    setError(null);
    try {
      onSaved(await resetModelAssignment(slot.slot));
      setEditing(false);
    } catch (caughtError) {
      setError(errorMessage(caughtError, "failed to reset the slot"));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="hierarchy-card">
      <div className="hierarchy-card-head">
        <span className="hierarchy-card-name">{slot.label}</span>
        {slot.needs.map((need) => (
          <span key={need} className="badge">
            {NEED_LABELS[need] ?? need}
          </span>
        ))}
      </div>
      <div className="model-slot-call-sites">{slot.call_sites}</div>
      {slot.warning && <p className="model-slot-warning">{slot.warning}</p>}

      {slot.assignment ? (
        <div className="model-slot-current">
          <span className="model-slot-model">{describeChoice(slot.assignment)}</span>
          <span className="model-slot-audit">
            Changed by {slot.assignment.updated_by_email ?? "a deleted user"},{" "}
            {new Date(slot.assignment.updated_at).toLocaleString()}
          </span>
        </div>
      ) : (
        <div className="model-slot-current">
          <span className="model-slot-model">Default: {defaultModel} · model&rsquo;s own reasoning default</span>
        </div>
      )}

      {!readOnly && !editing && (
        <div className="actions-cell">
          <button type="button" className="btn-text" onClick={startEditing}>
            Change
          </button>
          {slot.assignment && (
            <button type="button" className="btn-text" disabled={busy !== null} onClick={reset}>
              {busy === "resetting" ? "Resetting…" : "Reset to default"}
            </button>
          )}
        </div>
      )}

      {!readOnly && editing && (
        <div className="model-slot-editor">
          <ModelPicker
            slot={slot.slot}
            value={draft}
            defaultReasoningEffort={slot.default_reasoning_effort}
            onChange={(choice) => {
              setDraft(choice);
              setProbeResult(null);
            }}
          />
          <div className="actions-cell">
            <button type="button" className="btn btn-primary" disabled={!draft || busy !== null} onClick={save}>
              {busy === "saving" ? "Testing and saving…" : "Save"}
            </button>
            <button type="button" className="btn btn-ghost" disabled={!draft || busy !== null} onClick={test}>
              {busy === "testing" ? "Testing…" : "Test"}
            </button>
            <button type="button" className="btn-text" disabled={busy !== null} onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {error && <p className="alert model-slot-details" role="alert">{error}</p>}
      {probeResult && (
        <div className={`model-slot-probe ${probeResult.passed ? "model-slot-probe-passed" : "model-slot-probe-failed"}`}>
          <strong>{probeResult.passed ? "Probe passed" : "Probe failed"}</strong>
          <ul>
            {probeResult.details.map((detail) => (
              <li key={detail}>{detail}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// Settings section: the model and reasoning effort each node runs on (issue
// #24). An unset slot runs CHAT_MODEL at the model's own default. Saving
// runs the slot's real LangChain call against the model first and is
// refused, with the reason, if it fails. A save takes effect within 30
// seconds. Read-only without OPENROUTER_API_KEY.
export function ModelAssignmentsSection() {
  const [assignments, setAssignments] = useState<ModelAssignments | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    getModelAssignments()
      .then((fetchedAssignments) => {
        setAssignments(fetchedAssignments);
        setLoadError(null);
      })
      .catch((caughtError) => setLoadError(errorMessage(caughtError, "failed to load model settings")));
  }, []);

  return (
    <section className="settings-section">
      <h2>Models</h2>
      <p className="settings-section-lede">
        Choose the model and reasoning effort for each part of the pipeline. A slot left on the
        default runs the deployment&rsquo;s chat model. Saving first tests the model against that
        slot&rsquo;s real call and refuses it if the test fails; a saved change takes effect within
        30 seconds. Extraction changes only affect documents ingested afterwards.
      </p>

      {loadError && <p className="alert" role="alert">{loadError}</p>}
      {assignments && !assignments.gateway_configured && (
        <p className="model-slot-readonly">
          Read-only: OPENROUTER_API_KEY isn&rsquo;t set, so every slot runs the default model.
        </p>
      )}

      {assignments && (
        <div className="hierarchy-cards">
          {assignments.slots.map((slot) => (
            <SlotCard
              key={slot.slot}
              slot={slot}
              defaultModel={assignments.default_model}
              readOnly={!assignments.gateway_configured}
              onSaved={setAssignments}
            />
          ))}
        </div>
      )}
    </section>
  );
}

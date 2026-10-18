"use client";

// The Eval tab's per-slot model overrides for one rerun (issue #24): pick a
// model for any chat slot, or leave it on its production setting, then
// rerun. The overrides go with that one conversation only -- Settings
// aren't touched -- and the rerun's answer shows which models it ran on,
// to compare against the original answer in this row.
import { useState } from "react";
import type { ModelOverrides, ModelSlot } from "../../lib/model-assignments/types";
import { ModelPicker } from "../admin/model-picker";

export function RerunModelsPanel({
  slots,
  onRerun,
  onCancel,
}: {
  // The chat slots, as GET /admin/model-assignments lists them.
  slots: ModelSlot[];
  onRerun: (overrides: ModelOverrides) => void;
  onCancel: () => void;
}) {
  const [overrides, setOverrides] = useState<ModelOverrides>({});

  return (
    <div className="rerun-models">
      <p className="eval-detail-label">Rerun with models</p>
      {slots.map((slot) => (
        <div key={slot.slot} className="rerun-models-slot">
          <span className="rerun-models-label">
            {slot.label}
            <span className="rerun-models-current">
              {" "}
              (production: {slot.assignment ? slot.assignment.model_id : "default"})
            </span>
          </span>
          <ModelPicker
            slot={slot.slot}
            value={overrides[slot.slot] ?? null}
            defaultReasoningEffort={slot.default_reasoning_effort}
            unsetLabel="Production setting"
            onChange={(choice) =>
              setOverrides((previous) => {
                const next = { ...previous };
                if (choice) next[slot.slot] = choice;
                else delete next[slot.slot];
                return next;
              })
            }
          />
        </div>
      ))}
      <div className="actions-cell">
        <button
          type="button"
          className="btn btn-primary"
          disabled={Object.keys(overrides).length === 0}
          onClick={() => onRerun(overrides)}
        >
          Rerun
        </button>
        <button type="button" className="btn-text" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}

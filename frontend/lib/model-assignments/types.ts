// Mirrors backend/model_assignments/models.py (issue #24): which model, and
// which reasoning effort, each node that calls a chat model runs on.
export type ReasoningEffort = "none" | "low" | "medium" | "high";

export const REASONING_EFFORTS: ReasoningEffort[] = ["none", "low", "medium", "high"];

// The slots a chat turn runs (backend/model_assignments/slots.py's
// CHAT_SLOTS) -- the only ones the Eval tab's rerun can override.
export const CHAT_SLOTS = ["intent_classification", "deterministic_answer", "react_agent"] as const;

export interface ModelChoice {
  model_id: string;
  // null leaves the model's own default.
  reasoning_effort: ReasoningEffort | null;
}

export interface ModelAssignment extends ModelChoice {
  updated_by_email: string | null;
  updated_at: string;
}

export interface ModelSlot {
  slot: string;
  label: string;
  call_sites: string;
  needs: string[];
  default_reasoning_effort: ReasoningEffort;
  warning: string | null;
  // null while the slot runs the default.
  assignment: ModelAssignment | null;
}

export interface ModelAssignments {
  // False without OPENROUTER_API_KEY: everything is read-only and every
  // slot runs the default.
  gateway_configured: boolean;
  default_model: string;
  slots: ModelSlot[];
}

// One OpenRouter catalog entry whose metadata covers a slot's needs.
export interface CatalogModel {
  model_id: string;
  name: string;
  prompt_price_per_million: number | null;
  completion_price_per_million: number | null;
  supports_reasoning: boolean;
}

export interface ProbeResult {
  passed: boolean;
  details: string[];
}

// Slot name -> model, for one rerun conversation.
export type ModelOverrides = Record<string, ModelChoice>;

// What a slot actually ran on, as a rerun's "done" event reports it.
export interface SlotModelUsed {
  model_id: string;
  reasoning_effort: string | null;
  source: "default" | "setting" | "override";
}

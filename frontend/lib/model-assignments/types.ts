// Mirrors backend/model_assignments/models.py (issue #24): which model, and
// which reasoning effort, each node that calls a chat model runs on.
export type ReasoningEffort = "none" | "low" | "medium" | "high";

export const REASONING_EFFORTS: ReasoningEffort[] = ["none", "low", "medium", "high"];

// The slots a chat turn runs (backend/model_assignments/slots.py's
// CHAT_SLOTS) -- the only ones the Eval tab's rerun can override.
export const CHAT_SLOTS = ["intent_classification", "deterministic_answer", "react_agent"] as const;

// A chat model writes structured output; a decision model (issue #23,
// intent classification only) answers Choices with probabilities instead.
export type ModelKind = "chat" | "decision";

// A decision model's thresholds when none are given: backend
// model_assignments/slots.py's defaults. Starting points, to be tuned on
// the intent eval.
export const DEFAULT_FAMILY_CONFIDENCE_THRESHOLD = 0.6;
export const DEFAULT_QUESTION_TYPE_MARGIN = 0.15;

export interface ModelChoice {
  model_id: string;
  // null leaves the model's own default. Always null on a decision model.
  reasoning_effort: ReasoningEffort | null;
  // Omitted means "chat".
  model_kind?: ModelKind;
  // Decision models only: the family pick needs at least this probability,
  // or the rep is asked (0 < value <= 1).
  family_confidence_threshold?: number | null;
  // Decision models only: two question types within this of each other
  // both count, so the rep is asked which (0 <= value < 1).
  question_type_margin?: number | null;
}

export interface ModelAssignment extends ModelChoice {
  model_kind: ModelKind;
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
  // Whether the picker also offers decision models.
  accepts_decision_models: boolean;
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

// One OpenRouter catalog entry whose metadata covers a slot's needs, or a
// decision model the slot accepts (listed by its dated id).
export interface CatalogModel {
  model_id: string;
  name: string;
  prompt_price_per_million: number | null;
  completion_price_per_million: number | null;
  supports_reasoning: boolean;
  model_kind: ModelKind;
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
  // Present only when the slot ran a decision model.
  model_kind?: ModelKind;
  family_confidence_threshold?: number;
  question_type_margin?: number;
}

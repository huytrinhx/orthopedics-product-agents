import { request } from "../api/client";
import type { CatalogModel, ModelAssignments, ModelChoice, ProbeResult } from "./types";

// Mirrors backend/api/routes/model_assignments.py (issue #24). Admin-only.
const BASE = "/admin/model-assignments";

export async function getModelAssignments(): Promise<ModelAssignments> {
  return request(BASE);
}

// The picker's options for one slot: OpenRouter catalog models whose
// metadata covers what the slot needs, cached server-side for an hour.
export async function listSlotModels(slot: string): Promise<CatalogModel[]> {
  return request(`${BASE}/${slot}/models`);
}

// Runs the slot's live probe first; a model that fails it is rejected with
// the reason as the error message.
export async function saveModelAssignment(slot: string, choice: ModelChoice): Promise<ModelAssignments> {
  return request(`${BASE}/${slot}`, { method: "PUT", body: JSON.stringify(choice) });
}

// The same probe, without saving.
export async function testModelAssignment(slot: string, choice: ModelChoice): Promise<ProbeResult> {
  return request(`${BASE}/${slot}/test`, { method: "POST", body: JSON.stringify(choice) });
}

export async function resetModelAssignment(slot: string): Promise<ModelAssignments> {
  return request(`${BASE}/${slot}`, { method: "DELETE" });
}

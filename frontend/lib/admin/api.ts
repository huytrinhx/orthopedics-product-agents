import { request } from "../api/client";
import type { AdminSettings } from "./types";

export async function getAdminSettings(): Promise<AdminSettings> {
  return request("/admin/settings");
}

export async function setDefaultWorkflow(workflowName: string): Promise<AdminSettings> {
  return request("/admin/settings", {
    method: "PUT",
    body: JSON.stringify({ default_workflow: workflowName }),
  });
}

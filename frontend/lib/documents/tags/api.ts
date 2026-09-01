import { request } from "../../api/client";
import type { Tag } from "./types";

export async function listSystems(): Promise<Tag[]> {
  return request("/systems");
}

export async function createSystem(name: string): Promise<Tag> {
  return request("/systems", { method: "POST", body: JSON.stringify({ name }) });
}

export async function listDocumentTypes(): Promise<Tag[]> {
  return request("/document-types");
}

export async function createDocumentType(name: string): Promise<Tag> {
  return request("/document-types", { method: "POST", body: JSON.stringify({ name }) });
}

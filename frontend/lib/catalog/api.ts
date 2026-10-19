import { request } from "../api/client";
import type { FamilyDescription, ProductFamily, ProductFamilySummary, Tray } from "./types";

export async function listProductFamilies(): Promise<ProductFamily[]> {
  return request("/catalog/product-families");
}

// A blank display name resets it to the catalog name (server-side).
export async function setFamilyDisplayName(
  familyId: string,
  displayName: string
): Promise<ProductFamilySummary> {
  return request(`/catalog/product-families/${familyId}`, {
    method: "PATCH",
    body: JSON.stringify({ display_name: displayName }),
  });
}

export async function setTrayDisplayName(trayId: string, displayName: string): Promise<Tray> {
  return request(`/catalog/trays/${trayId}`, {
    method: "PATCH",
    body: JSON.stringify({ display_name: displayName }),
  });
}

// A blank description goes back to the generated one (server-side).
export async function setFamilyDescription(familyId: string, description: string): Promise<FamilyDescription> {
  return request(`/catalog/product-families/${familyId}/description`, {
    method: "PATCH",
    body: JSON.stringify({ description }),
  });
}

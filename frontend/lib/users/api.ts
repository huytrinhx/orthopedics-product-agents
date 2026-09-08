import { request } from "../api/client";
import type { AuthUser } from "../auth/types";

export async function listUsers(): Promise<AuthUser[]> {
  return request("/users/");
}

export async function setUserActive(userId: string, isActive: boolean): Promise<AuthUser> {
  return request(`/users/${userId}/active`, {
    method: "PATCH",
    body: JSON.stringify({ is_active: isActive }),
  });
}

import { request } from "../api/client";
import type { AuthUser, TokenResponse } from "./types";

export async function signup(email: string, password: string): Promise<TokenResponse> {
  return request("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  return request("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function getCurrentUser(): Promise<AuthUser> {
  return request("/auth/me");
}

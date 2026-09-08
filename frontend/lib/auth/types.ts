export interface AuthUser {
  id: string;
  email: string;
  is_admin: boolean;
  // Non-admin only -- an admin's is_active is never consulted (backend/auth/
  // dependencies.py's require_chat_access bypasses the check for them
  // entirely), so it's always true for an admin row. New signups start
  // false and need an admin to flip them on via the Users tab.
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  user: AuthUser;
}

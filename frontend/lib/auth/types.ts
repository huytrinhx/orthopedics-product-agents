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
  // Stamped on signup and every later login/OAuth callback (backend/auth/
  // repository.py's touch_last_login/promote_to_admin) -- null is only
  // possible for a row that predates the last_login_at column.
  last_login_at: string | null;
}

export interface TokenResponse {
  access_token: string;
  user: AuthUser;
}

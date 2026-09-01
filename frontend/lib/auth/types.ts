export interface AuthUser {
  id: string;
  email: string;
  is_admin: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  user: AuthUser;
}

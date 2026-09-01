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

export type DocumentStatus = "pending" | "queued" | "processing" | "done" | "failed";

export interface Tag {
  id: string;
  name: string;
}

export interface DocumentRecord {
  id: string;
  filename: string;
  status: DocumentStatus;
  error: string | null;
  uploaded_by: string;
  created_at: string;
  updated_at: string;
  system: Tag | null;
  document_type: Tag | null;
}

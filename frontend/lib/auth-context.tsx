"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { getCurrentUser, login as apiLogin, signup as apiSignup } from "./api";
import { clearToken, getToken, setToken } from "./auth";
import type { AuthUser } from "./types";

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // A Google OAuth redirect (backend/api/routes/auth.py's google_callback)
    // carries a fresh token in the URL, since a full-page redirect has no
    // other way to hand a static-export SPA a session.
    const params = new URLSearchParams(window.location.search);
    const tokenFromUrl = params.get("auth_token");
    if (tokenFromUrl) {
      setToken(tokenFromUrl);
      window.history.replaceState({}, "", window.location.pathname);
    }

    if (!getToken()) {
      setLoading(false);
      return;
    }
    getCurrentUser()
      .then(setUser)
      .catch(() => clearToken())
      .finally(() => setLoading(false));
  }, []);

  async function login(email: string, password: string) {
    const { access_token, user } = await apiLogin(email, password);
    setToken(access_token);
    setUser(user);
  }

  async function signup(email: string, password: string) {
    const { access_token, user } = await apiSignup(email, password);
    setToken(access_token);
    setUser(user);
  }

  function logout() {
    clearToken();
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}

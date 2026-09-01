"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";
import { API_BASE } from "../../lib/api";
import { useAuth } from "../../lib/auth-context";

export default function LoginPage() {
  const { login } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(email, password);
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "login failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="page-narrow">
      <h1>Log in</h1>
      <div className="card">
        <form onSubmit={handleSubmit}>
          <label className="field">
            Email
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoFocus
            />
          </label>
          <label className="field">
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>
          {error && <p className="alert" role="alert">{error}</p>}
          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? "Logging in…" : "Log in"}
          </button>
        </form>
        <p className="divider-link">
          <a href={`${API_BASE}/auth/google/login`}>Sign in with Google</a>
        </p>
      </div>
      <p className="divider-link">
        No account? <Link href="/signup">Sign up</Link>
      </p>
    </main>
  );
}

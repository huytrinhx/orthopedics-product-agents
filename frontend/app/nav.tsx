"use client";

import Link from "next/link";
import { useAuth } from "../lib/auth-context";

export function Nav() {
  const { user, loading, logout } = useAuth();

  return (
    <nav className="nav">
      <div className="nav-inner">
        <Link href="/" className="logo">
          Ortho<em>Mate</em>
        </Link>

        {!loading && user && (
          <>
            <div className="nav-links">
              <Link href="/chat">Chat</Link>
              {user.is_admin && (
                <>
                  <Link href="/documents">Documents</Link>
                  <Link href="/evals">Evals</Link>
                </>
              )}
            </div>
            <div className="nav-right">
              {user.is_admin && <span className="admin-tag">Admin</span>}
              <span className="email">{user.email}</span>
              <button type="button" className="btn-text" onClick={logout}>
                Log out
              </button>
            </div>
          </>
        )}

        {!loading && !user && (
          <div className="nav-links">
            <Link href="/login">Log in</Link>
            <Link href="/signup">Sign up</Link>
          </div>
        )}
      </div>
    </nav>
  );
}

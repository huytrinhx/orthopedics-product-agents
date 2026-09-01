"use client";

import Link from "next/link";
import { useAuth } from "../lib/auth-context";

export default function Home() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <main className="page">
        <p>Loading…</p>
      </main>
    );
  }

  if (!user) {
    return (
      <main className="page">
        <span className="eyebrow">Foot &amp; ankle · REFLEX + MIS</span>
        <h1>Cited answers for the questions that come up mid-case.</h1>
        <p className="lede">
          OrthoMate answers from Medline&rsquo;s own surgical technique guides, IFUs, and
          brochures — every claim traceable to the page it came from. Log in to ask.
        </p>
      </main>
    );
  }

  return (
    <main className="page">
      <span className="eyebrow">Signed in as {user.email}</span>
      <h1>What do you need?</h1>
      <div className="link-grid">
        <Link href="/chat" className="link-card">
          <div>
            <div className="title">Chat</div>
            <p style={{ margin: "4px 0 0", color: "var(--ink-soft)", fontSize: "0.9rem" }}>
              Ask a question and get a cited answer.
            </p>
          </div>
          <span className="arrow">→</span>
        </Link>
        {user.is_admin && (
          <>
            <Link href="/documents" className="link-card">
              <div>
                <div className="title">Document Manager</div>
                <p style={{ margin: "4px 0 0", color: "var(--ink-soft)", fontSize: "0.9rem" }}>
                  Upload and track the documents OrthoMate answers from.
                </p>
              </div>
              <span className="arrow">→</span>
            </Link>
            <Link href="/evals" className="link-card">
              <div>
                <div className="title">Evals</div>
                <p style={{ margin: "4px 0 0", color: "var(--ink-soft)", fontSize: "0.9rem" }}>
                  Review answer quality against the golden dataset.
                </p>
              </div>
              <span className="arrow">→</span>
            </Link>
          </>
        )}
      </div>
    </main>
  );
}

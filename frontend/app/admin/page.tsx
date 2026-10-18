"use client";

import Link from "next/link";
import { useAuth } from "../../lib/auth-context";
import { DocTypeHierarchySection } from "./doc-type-hierarchy-section";
import { DocumentTypesSection } from "./document-types-section";
import { ModelAssignmentsSection } from "./model-assignments-section";
import { ProductFamiliesSection } from "./product-families-section";
import { WorkflowSection } from "./workflow-section";

// The Settings page (nav: "Settings"). Each concern is its own
// self-contained section component in this folder -- it loads and saves its
// own data -- so a new setting is one new <name>-section.tsx file plus one
// line below, without touching the others.
export default function AdminPage() {
  const { user, loading } = useAuth();

  if (loading) return <main className="page-wide"><p>Loading…</p></main>;

  if (!user || !user.is_admin) {
    return (
      <main className="page-wide">
        <h1>Settings</h1>
        <p>This page is admin-only.</p>
        <Link href="/">Back home</Link>
      </main>
    );
  }

  return (
    <main className="page-wide">
      <span className="eyebrow">Admin</span>
      <h1>Settings</h1>

      <WorkflowSection />
      <ModelAssignmentsSection />
      <DocumentTypesSection />
      <ProductFamiliesSection />
      <DocTypeHierarchySection />
    </main>
  );
}

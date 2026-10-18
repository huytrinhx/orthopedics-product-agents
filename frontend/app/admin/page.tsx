"use client";

import { useEffect, useState, type ComponentType } from "react";
import Link from "next/link";
import { useAuth } from "../../lib/auth-context";
import { DocTypeHierarchySection } from "./doc-type-hierarchy-section";
import { DocumentTypesSection } from "./document-types-section";
import { ModelAssignmentsSection } from "./model-assignments-section";
import { ProductFamiliesSection } from "./product-families-section";
import { SettingsNav } from "./settings-nav";
import { WorkflowSection } from "./workflow-section";

// The Settings page (nav: "Settings"). Each concern is its own
// self-contained section component in this folder -- it loads and saves its
// own data -- so a new setting is one new <name>-section.tsx file plus one
// entry below, without touching the others.
//
// Only the selected section renders, chosen from the vertical SettingsNav.
// The choice lives in the URL as `?section=<id>` so a link or a reload lands
// on it; an unknown or missing id means the first section. It's read from
// window.location rather than useSearchParams, which a static export would
// force behind a Suspense boundary (same reasoning as /chat's `?ask=`).
const SECTIONS: { id: string; label: string; Component: ComponentType }[] = [
  { id: "workflow", label: "Workflow", Component: WorkflowSection },
  { id: "models", label: "Models", Component: ModelAssignmentsSection },
  { id: "document-types", label: "Document types", Component: DocumentTypesSection },
  { id: "families", label: "Families & trays", Component: ProductFamiliesSection },
  { id: "doc-type-hierarchy", label: "Doc type hierarchy", Component: DocTypeHierarchySection },
];

function sectionIdFromUrl(): string {
  const requestedId = new URLSearchParams(window.location.search).get("section");
  const requestedSection = SECTIONS.find((section) => section.id === requestedId);
  return (requestedSection ?? SECTIONS[0]).id;
}

export default function AdminPage() {
  const { user, loading } = useAuth();
  const [selectedId, setSelectedId] = useState(SECTIONS[0].id);

  useEffect(() => {
    setSelectedId(sectionIdFromUrl());
    const followHistory = () => setSelectedId(sectionIdFromUrl());
    window.addEventListener("popstate", followHistory);
    return () => window.removeEventListener("popstate", followHistory);
  }, []);

  function selectSection(id: string) {
    if (id === selectedId) return;
    const url = new URL(window.location.href);
    url.searchParams.set("section", id);
    window.history.pushState(null, "", url);
    setSelectedId(id);
  }

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

  const selectedSection = SECTIONS.find((section) => section.id === selectedId) ?? SECTIONS[0];
  const SelectedComponent = selectedSection.Component;

  return (
    <main className="page-wide">
      <span className="eyebrow">Admin</span>
      <h1>Settings</h1>

      <div className="settings-layout">
        <SettingsNav
          items={SECTIONS.map(({ id, label }) => ({ id, label }))}
          selectedId={selectedSection.id}
          onSelect={selectSection}
        />
        <div className="settings-content">
          <SelectedComponent key={selectedSection.id} />
        </div>
      </div>
    </main>
  );
}

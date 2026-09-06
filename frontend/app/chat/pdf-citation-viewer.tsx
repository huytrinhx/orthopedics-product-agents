"use client";

// Ticket 26: renders the original PDF at a cited chunk's page, then does a
// best-effort in-page text search (pdf.js's own text layer, not a second
// extraction) to scroll/highlight the matched passage. Page-jump is
// guaranteed (page_number is stored at ingestion); the text match is
// best-effort because pdfplumber's extraction (ingestion time) and pdf.js's
// text layer (render time) don't always tokenize identically -- normalized,
// first-~80-characters matching keeps that gap from mattering in practice.
import { useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// Self-hosted (frontend/public/pdf.worker.min.mjs, copied from
// pdfjs-dist/build at the version react-pdf depends on) rather than a CDN --
// this app has no other runtime dependency on an external host.
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

// Search-anchor length matches the ticket's design: the start of the cited
// passage, not the full chunk (a full-length match is increasingly likely to
// miss on line-wrap/hyphenation differences between the two extractions).
const SEARCH_ANCHOR_LENGTH = 80;

function normalize(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}

interface PdfCitationViewerProps {
  file: { data: Uint8Array };
  pageNumber: number;
  searchText: string;
  onError: () => void;
}

export function PdfCitationViewer({ file, pageNumber, searchText, onError }: PdfCitationViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const highlightedRef = useRef<HTMLElement[]>([]);
  const [containerWidth, setContainerWidth] = useState(360);
  const [textLayerRenderCount, setTextLayerRenderCount] = useState(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) setContainerWidth(Math.max(200, Math.floor(width)));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Re-runs on a fresh text-layer render (a new page) and whenever the
  // cited chunk changes on the *same* already-rendered page (clicking a
  // different citation into the same page shouldn't require a remount to
  // re-search).
  useEffect(() => {
    if (textLayerRenderCount === 0) return;
    const container = containerRef.current;
    highlightedRef.current.forEach((el) => el.classList.remove("chat-pdf-highlight"));
    highlightedRef.current = [];
    if (!container) return;

    const spans = Array.from(
      container.querySelectorAll<HTMLSpanElement>(".react-pdf__Page__textContent span")
    );
    if (spans.length === 0) return;

    const parts: { el: HTMLSpanElement; start: number; end: number }[] = [];
    let hay = "";
    for (const el of spans) {
      const norm = normalize(el.textContent ?? "");
      if (!norm) continue;
      const start = hay.length === 0 ? 0 : hay.length + 1;
      hay = hay.length === 0 ? norm : `${hay} ${norm}`;
      parts.push({ el, start, end: hay.length });
    }

    const needle = normalize(searchText).slice(0, SEARCH_ANCHOR_LENGTH);
    if (!needle) return;
    const idx = hay.indexOf(needle);
    if (idx === -1) return; // falls back to wherever the page itself scrolled to (its top)

    const matchEnd = idx + needle.length;
    const matched = parts.filter((p) => p.end > idx && p.start < matchEnd);
    matched.forEach((p) => p.el.classList.add("chat-pdf-highlight"));
    highlightedRef.current = matched.map((p) => p.el);
    matched[0]?.el.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [searchText, textLayerRenderCount]);

  return (
    <div ref={containerRef} className="chat-pdf-viewer">
      <Document
        file={file}
        onLoadError={onError}
        loading={<div className="empty-state">Loading PDF…</div>}
      >
        <Page
          key={pageNumber}
          pageNumber={pageNumber}
          width={containerWidth}
          onRenderTextLayerSuccess={() => setTextLayerRenderCount((n) => n + 1)}
          onRenderError={onError}
        />
      </Document>
    </div>
  );
}

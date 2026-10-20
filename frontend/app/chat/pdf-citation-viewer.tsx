"use client";

// Renders the whole cited PDF as one scrolling column, opened at the cited
// chunk's page, then does a best-effort text search on that page (pdf.js's
// own text layer, not a second extraction) to highlight the passage.
// Page-jump is guaranteed (page_number is stored at ingestion, ticket 26);
// the text match is best-effort because pdfplumber's extraction (ingestion
// time) and pdf.js's text layer (render time) don't always tokenize
// identically -- normalized, first-~80-characters matching keeps that gap
// from mattering in practice.
//
// Every page gets a placeholder sized from its own aspect ratio, so the
// scrollbar is the whole document's from the start, but only the pages near
// the visible area are actually drawn (an IntersectionObserver over the
// placeholders): drawing all forty pages of a technique guide at once is
// what would make scrolling and clicking stutter.
//
// The scroll container keeps a stable scrollbar gutter, so content growing
// past the viewport never changes the page width -- the side pane this
// replaced re-rendered every page whenever its scrollbar came or went.
// The cited page scrolls into view, and the highlight is scrolled to, once
// per citation: a page drawn again after the rep scrolls back to it gets
// its highlight back without yanking the view.
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Document, Page, pdfjs, type DocumentProps } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// Self-hosted (frontend/public/pdf.worker.min.mjs, copied from
// pdfjs-dist/build at the version react-pdf depends on) rather than a CDN --
// this app has no other runtime dependency on an external host.
pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

type LoadedPdf = Parameters<NonNullable<DocumentProps["onLoadSuccess"]>>[0];

// Search-anchor length matches the ticket's design: the start of the cited
// passage, not the full chunk (a full-length match is increasingly likely to
// miss on line-wrap/hyphenation differences between the two extractions).
const SEARCH_ANCHOR_LENGTH = 80;
// How far beyond the visible area pages are drawn ahead of the scroll.
const RENDER_MARGIN = "1200px 0px";
const PAGE_GAP = 12;
// Width changes smaller than this (a sub-pixel resize) don't redraw pages.
const WIDTH_STEP = 16;

function normalize(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}

function highlightPassage(pageElement: HTMLElement, searchText: string): HTMLElement | null {
  const spans = Array.from(
    pageElement.querySelectorAll<HTMLSpanElement>(".react-pdf__Page__textContent span")
  );
  const parts: { element: HTMLSpanElement; start: number; end: number }[] = [];
  let haystack = "";
  for (const element of spans) {
    const normalized = normalize(element.textContent ?? "");
    if (!normalized) continue;
    const start = haystack.length === 0 ? 0 : haystack.length + 1;
    haystack = haystack.length === 0 ? normalized : `${haystack} ${normalized}`;
    parts.push({ element, start, end: haystack.length });
  }
  const needle = normalize(searchText).slice(0, SEARCH_ANCHOR_LENGTH);
  if (!needle) return null;
  const matchIndex = haystack.indexOf(needle);
  if (matchIndex === -1) return null;
  const matchEnd = matchIndex + needle.length;
  const matched = parts.filter((part) => part.end > matchIndex && part.start < matchEnd);
  matched.forEach((part) => part.element.classList.add("chat-pdf-highlight"));
  return matched[0]?.element ?? null;
}

interface PdfCitationViewerProps {
  file: { data: Uint8Array };
  pageNumber: number;
  searchText: string;
  onError: () => void;
}

export function PdfCitationViewer({ file, pageNumber, searchText, onError }: PdfCitationViewerProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const pageElementsRef = useRef(new Map<number, HTMLDivElement>());
  const [width, setWidth] = useState(0);
  // Height / width of each page at scale 1, index 0 = page 1.
  const [aspectRatios, setAspectRatios] = useState<number[] | null>(null);
  const [nearbyPages, setNearbyPages] = useState<Set<number>>(() => new Set());
  // The citation ("page|text") the view was last scrolled for, so it only
  // happens once per citation.
  const scrolledToPageRef = useRef<string | null>(null);
  const scrolledToPassageRef = useRef<string | null>(null);
  const citationKey = `${pageNumber}|${searchText}`;
  const pagesLaidOut = aspectRatios !== null && width > 0;

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      const next = Math.floor(element.clientWidth / WIDTH_STEP) * WIDTH_STEP - 2 * PAGE_GAP;
      setWidth((previous) => (Math.abs(previous - next) >= WIDTH_STEP ? Math.max(200, next) : previous));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const root = scrollRef.current;
    if (!root || !pagesLaidOut) return;
    const observer = new IntersectionObserver(
      (entries) => {
        setNearbyPages((previous) => {
          const next = new Set(previous);
          for (const entry of entries) {
            const page = Number((entry.target as HTMLElement).dataset.pageNumber);
            if (entry.isIntersecting) next.add(page);
            else next.delete(page);
          }
          return next;
        });
      },
      { root, rootMargin: RENDER_MARGIN }
    );
    pageElementsRef.current.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, [pagesLaidOut]);

  // Jump (not smooth-scroll) to the cited page as soon as the placeholders
  // have their heights, before any page is drawn.
  useLayoutEffect(() => {
    if (!pagesLaidOut || scrolledToPageRef.current === citationKey) return;
    const target = pageElementsRef.current.get(pageNumber);
    const root = scrollRef.current;
    if (!target || !root) return;
    // .citation-viewer-pages is position: relative, so offsetTop is
    // measured from the top of the scrolled content.
    root.scrollTop = target.offsetTop - PAGE_GAP;
    scrolledToPageRef.current = citationKey;
  }, [pagesLaidOut, pageNumber, citationKey]);

  async function handleLoadSuccess(pdf: LoadedPdf) {
    const pages = await Promise.all(
      Array.from({ length: pdf.numPages }, (_, index) => pdf.getPage(index + 1))
    );
    setAspectRatios(
      pages.map((page) => {
        const viewport = page.getViewport({ scale: 1 });
        return viewport.height / viewport.width;
      })
    );
  }

  function handleTextLayerRendered(page: number) {
    if (page !== pageNumber) return;
    const pageElement = pageElementsRef.current.get(page);
    if (!pageElement) return;
    const firstMatch = highlightPassage(pageElement, searchText);
    if (firstMatch && scrolledToPassageRef.current !== citationKey) {
      scrolledToPassageRef.current = citationKey;
      firstMatch.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  return (
    <div ref={scrollRef} className="citation-viewer-pages">
      <Document
        file={file}
        onLoadSuccess={handleLoadSuccess}
        onLoadError={onError}
        loading={<div className="empty-state">Loading PDF…</div>}
      >
        {pagesLaidOut &&
          aspectRatios.map((aspectRatio, index) => {
            const page = index + 1;
            return (
              <div
                key={page}
                data-page-number={page}
                ref={(element) => {
                  if (element) pageElementsRef.current.set(page, element);
                  else pageElementsRef.current.delete(page);
                }}
                className="citation-viewer-page"
                style={{ width, height: Math.round(width * aspectRatio) }}
              >
                {(nearbyPages.has(page) || page === pageNumber) && (
                  <Page
                    pageNumber={page}
                    width={width}
                    loading={null}
                    onRenderTextLayerSuccess={() => handleTextLayerRendered(page)}
                    onRenderError={onError}
                  />
                )}
              </div>
            );
          })}
      </Document>
    </div>
  );
}

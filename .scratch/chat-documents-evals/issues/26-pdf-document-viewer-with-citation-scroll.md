# 26: Document viewer -- render the original PDF, auto-scroll to the cited paragraph

**What to build:** Clicking a citation chip in chat currently opens a side panel showing plain *extracted* text (`GET /documents/{id}/chunks`, `frontend/app/chat/page.tsx`'s `openCitationPane`) -- never the original file. Replace that panel's content with the actual rendered PDF, scrolled and highlighted to the cited passage, for real page-accurate provenance instead of a re-flowed text approximation.

Requested directly (2026-09-05), scoped via a grilling session -- captured here for traceability, matching this session's convention for tickets not pre-scoped in the original 17.

**Blocked by:** 06 (Ingestion vector leg, done -- this changes its chunk schema), 10 (Chat history sidebar, done -- citations already resolve through `chat_threads`/`ChatCitationOut`).

**Status:** ready-for-agent

## Ground truth (found live, 2026-09-05)

- `documents/service.py`/`ingestion/text_extraction.py`'s `_extract_pdf` joins every PDF page's text into one string (`"\n\n".join(pages)`) before `chunking.py` ever sees it -- page boundaries are discarded today. There is currently no way to know which page any given chunk came from.
- No endpoint serves the raw PDF file at all -- everything downstream of ingestion works from extracted text/chunks only. `documents/repository.py`'s `storage_path` still points at the file on disk, so it's there, just never exposed.
- `GET /documents/{id}/chunks` is any-authenticated-user accessible (not admin-gated) -- it backs the rep-facing chat citation panel, so whatever replaces/extends it needs the same access level.
- A bare `<iframe>`/native browser PDF viewer can only jump to a page number via a URL fragment (`#page=N`) -- it has no API for scrolling to specific text. Getting to paragraph-level requires a real PDF.js-based renderer with its own text layer and programmatic scroll control (`react-pdf` is the standard choice for this in a React app).

## Design (settled via grilling)

1. **Replaces the existing side panel in place** -- no new standalone route. The panel already has the right wiring (opens on citation click, scoped to one document, highlights the "active" item); only its *content* changes from a chunk-text list to a rendered PDF.
2. **Page-accuracy is real, paragraph-accuracy is best-effort ("hybrid")**:
   - Each chunk gets a `page_number` (the page its text **starts** on -- a chunk that spans a page boundary is *not* split at the boundary; that would touch chunking behavior already tuned for retrieval quality this session, for no real benefit to this feature). Requires: `text_extraction.py`'s PDF path returns per-page text (not one joined string) so `chunking.py` can attribute a starting page to each chunk; a new nullable `page_number` column on `chunks` (nullable because non-PDF chunks and pre-existing chunks from before this ships have none).
   - Clicking a citation always jumps to the right page (guaranteed, from the stored `page_number`).
   - On top of that, the frontend searches pdf.js's text layer **within that one page only** for the chunk's own stored text, normalized, and scrolls/highlights to the match if found -- falls back to the top of the page if no confident match. Search anchor is the chunk's **first ~80 normalized characters**, not the full chunk -- a full-length match is increasingly likely to miss on line-wrap/hyphenation differences between `pdfplumber`'s extraction (used at ingestion) and pdf.js's own text layer (used at render time), and the start of the passage is what "scroll to that paragraph" actually means.
3. **Non-PDF documents (.txt/.md) are unchanged** -- the panel keeps today's plain-text chunk display for those; this feature only replaces the panel's content when the cited document's file extension is `.pdf`.
4. **No backfill migration.** Documents indexed before this ships just have `page_number: null` on their existing chunks -- the viewer falls back to opening at page 1 for those until someone re-indexes (a one-click admin action that already exists in the Document Manager). Nothing forces a mass re-index.
5. **Highlight treatment matches the existing convention** -- the current panel already highlights the "active" chunk (`chat-doc-chunk-active`); apply the same visual idea to the matched text in the rendered PDF once found, don't invent a new highlight language.
6. **Render failure is a plain error message**, no fallback-to-text toggle -- keep this scoped until there's a real failure mode to design around.

## Acceptance criteria

- [ ] Migration: `chunks.page_number` (integer, nullable).
- [ ] `ingestion/text_extraction.py`: PDF extraction returns per-page text (not pre-joined) so `ingestion/chunking.py` can attribute each chunk to its starting page; heading detection (font-size heuristic) still works the same, just needs to track which page it's currently on.
- [ ] `ingestion/chunking.py`: `chunk_document`'s output includes `page_number` per chunk (the page the chunk's content starts on).
- [ ] `retrieval/vector_store.py`/`documents/repository.py`: `page_number` flows through to `ChunkOut`/`GET /documents/{id}/chunks`'s response.
- [ ] New backend route serving the raw file bytes for a document (e.g. `GET /documents/{id}/file`, `Content-Type` matching the stored file, same any-authenticated-user access level as `/chunks` -- not admin-gated) so the frontend can fetch it (with auth headers -- a bare `<iframe src>`/`img src` can't attach an Authorization header, so this is a `fetch()` + Blob handed to `react-pdf`, not a direct URL).
- [ ] Frontend: `react-pdf` (or equivalent pdf.js wrapper) added, citation panel renders the fetched PDF at the cited chunk's `page_number`, then searches that page's text layer for the chunk's first ~80 normalized characters and scrolls/highlights to the match (falling back to page-top on no match).
- [ ] Citations into a `.txt`/`.md` document, or a chunk with `page_number: null` (pre-existing, not yet re-indexed), both degrade gracefully -- verified explicitly, not just assumed: the former keeps the current text panel, the latter opens the PDF at page 1 rather than erroring.
- [ ] Verified live against a real indexed PDF (re-index one of the existing MIS Foot Recon documents to get real `page_number`s) -- click a citation in real chat, confirm the panel opens the correct page and lands on/near the correct paragraph, not just that a page number is stored correctly in the DB.

**Explicitly out of scope:** a standalone document-viewer route independent of the chat citation flow; backfilling `page_number` for already-indexed documents; any change to non-PDF document handling.

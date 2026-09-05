import type { ChatCitation } from "./types";

// Shared by app/chat/page.tsx and app/evals/ (ticket 15's rerun view, which
// embeds the same live conversation UI) -- extracted here so both stay
// pixel/behavior-identical rather than drifting two copies of "what a chat
// turn's status/citations look like."
export const STATUS_LABELS: Record<string, string> = {
  detect_intent: "Understanding your question…",
  resolve_synonyms: "Checking terminology…",
  hybrid_retrieve: "Searching documents…",
  rerank: "Ranking results…",
  generate: "Writing answer…",
  self_eval: "Checking answer quality…",
  request_clarification: "Preparing a follow-up question…",
  finalize: "Finishing up…",
};

// Mirrors backend/agents/citations.py's _CITATION_PATTERN exactly -- the
// model writes these `[document-id#chunk-index]` markers inline as its
// citation convention, but a raw Postgres UUID means nothing to a rep
// reading the answer. The chat-citations chips below the bubble are the
// actual clickable source list, so the inline marker is just stripped from
// what's rendered, not shown as literal bracketed text.
const _CITATION_DISPLAY_PATTERN =
  /\s?\[[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}#\d+\]/g;

export function stripCitationMarkers(content: string): string {
  return content.replace(_CITATION_DISPLAY_PATTERN, "");
}

// Two citations into the same document previously rendered as identical
// chips (filename only) -- a rep can't tell "the dosing table" apart from
// "the contraindications section" of the same PDF without clicking both.
// Falls back to the chunk's position when it has no heading of its own.
export function citationLabel(c: ChatCitation): string {
  return c.section_title ? `${c.filename} — ${c.section_title}` : `${c.filename} (#${c.chunk_index + 1})`;
}

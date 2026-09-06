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
  resolve_skus: "Matching part numbers…",
  aggregate_facts: "Gathering catalog facts…",
  generate: "Writing answer…",
  self_eval: "Checking answer quality…",
  request_clarification: "Preparing a follow-up question…",
  finalize: "Finishing up…",
};

// Mirrors backend/agents/citations.py's _CITATION_GROUP_PATTERN exactly --
// the model writes these `[document-id#chunk-index]` markers inline as its
// citation convention, but a raw Postgres UUID means nothing to a rep
// reading the answer. The chat-citations chips below the bubble are the
// actual clickable source list, so the inline marker is just stripped from
// what's rendered, not shown as literal bracketed text.
//
// The trailing `(?:\s*,\s*...)*` group matters: found live (2026-09-05)
// that the model sometimes cites two sources for one claim as a single
// bracket with a comma between them ("[id-a#3, id-b#5]") rather than two
// separate brackets -- the original single-ref pattern left that whole
// bracket completely unmatched, so the raw UUID marker leaked into the
// rendered answer verbatim. Matches one-or-more comma-separated refs inside
// one bracket so every shape the model actually produces gets stripped.
const _CITATION_REF = "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}#\\d+";
const _CITATION_DISPLAY_PATTERN = new RegExp(
  `\\s?\\[${_CITATION_REF}(?:\\s*,\\s*${_CITATION_REF})*\\]`,
  "g"
);

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

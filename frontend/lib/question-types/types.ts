export interface RankedDocumentType {
  id: string;
  name: string;
}

// A question type is code (backend/agents/question_types.py); only its
// ranked document types are data an admin edits (issue #17).
export interface QuestionType {
  name: string;
  description: string;
  // False only for `unclassified`, which always retrieves unfiltered.
  editable: boolean;
  // Most authoritative first. Empty means no doc-type filter.
  document_types: RankedDocumentType[];
}

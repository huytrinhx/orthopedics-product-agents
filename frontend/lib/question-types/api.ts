import { request } from "../api/client";
import type { QuestionType } from "./types";

export async function listQuestionTypes(): Promise<QuestionType[]> {
  return request("/question-types");
}

// Replaces the question type's whole ranked list (most authoritative
// first) and resolves to the question type as saved.
export async function setQuestionTypeDocumentTypes(
  questionTypeName: string,
  documentTypeIds: string[]
): Promise<QuestionType> {
  return request(`/question-types/${questionTypeName}/document-types`, {
    method: "PUT",
    body: JSON.stringify({ document_type_ids: documentTypeIds }),
  });
}

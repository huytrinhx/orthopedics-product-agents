# Golden Datasets

Generated from `feedback-notes.csv` by `build_dataset.py` (rerun it after new
rows land in the CSV; don't hand-edit the JSONL):

- One QA file per Product Family (`mis-foot-recon-system.jsonl`, ...):
  `query`, `expected_answer`, `expected_citations`, `product_family`,
  `question_type`, plus the reviewers' feedback.
- `intent_detection.jsonl`: `turns`, `expected_product_family`,
  `expected_question_type`, `expects_family_clarification`,
  `expects_question_type_clarification`.

`intent_detection.jsonl` is scored by `python -m evals.harness intent`
(`backend/evals/intent_eval.py`, issue #25 Slice 1). Scoring the QA files
against a workflow with the shared judge (`backend/agents/judge.py`) is
Slice 2 and isn't built yet.

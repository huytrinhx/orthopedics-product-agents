# Golden Datasets

Drop golden evaluation datasets here as JSONL, one example per line:

```json
{"query": "...", "expected_answer": "...", "expected_citations": ["doc-id#chunk-id"]}
```

Consumed by `backend/evals/harness.py`, which runs a dataset against any
workflow registered in `backend/agents/registry.py` and scores results with
the shared judge (`backend/agents/judge.py`) on the same four axes users
rate in the feedback UI: faithfulness, relevance, style, citation.

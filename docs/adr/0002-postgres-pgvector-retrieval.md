# 0002 — Retrieval on Postgres/pgvector, not Azure AI Search

## Status

Accepted

## Context

The scaffold used Azure AI Search for hybrid (vector + BM25) retrieval,
authenticated via the same enterprise-tenant service principal as Azure
OpenAI, with Neo4j's synonym graph synced into AI Search's synonym-map
feature. Once deployment moved off AKS to Railway (ADR 0001), staying on
Azure AI Search was still technically possible — it's reachable over the
network from anywhere — but kept an Azure dependency and its
enterprise-tenant auth path in a stack that was otherwise leaving Azure.
Every retrieval call in the code was still a stub (`NotImplementedError`)
at the time of this decision, so there was no working logic being thrown
away.

## Decision

Move the vector index into the same Postgres instance already used for the
checkpointer/store/metadata/feedback/eval-results roles, using the
pgvector extension. The keyword leg of hybrid search becomes Postgres
full-text search (`tsvector`/`ts_rank`) instead of AI Search's BM25.
Synonym expansion was already designed to query the Neo4j/AuraDB graph
directly (`backend/agents/tools/synonym_resolve.py`), not through a synced
index, so it's unaffected — `sync_synonym_map`-style logic (pushing Neo4j
synonyms into an external index) is simply gone, not replaced.

## Consequences

- One fewer external managed service (Azure AI Search) and one fewer
  credential to provision.
- Postgres now has vector, full-text, and transactional data in one
  instance — simpler ops, but means Postgres capacity/perf now matters for
  retrieval latency too, not just app metadata.
- No native synonym-aware BM25 index the way AI Search's synonym maps
  provided — keyword matching relies on Postgres full-text search
  correctly seeing synonym-expanded terms rather than an index doing the
  expansion server-side. `backend/retrieval/vector_store.py` is where this
  tradeoff is implemented.

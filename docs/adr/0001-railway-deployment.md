# 0001 — Deploy on Railway, not AKS

## Status

Accepted

## Context

The repo was scaffolded with a full Azure-native deployment path: Bicep
modules for AKS, Azure AI Search, Postgres Flexible Server, and ACR, plus a
GitHub Actions workflow (`deploy.yml`) that built images and pushed them to
AKS. At the same time, the README's own architecture-decision list
contained a contradicting line claiming the deployment target was
"Supabase + Railway" — a leftover from drafting against a different
project's (fhir-bridge) README as a template. Nothing had actually been
deployed yet; this is an early-stage scaffold with every retrieval/LLM call
still a stub.

## Decision

Deploy as a single Railway service instead of AKS. Retire
`infra/bicep/*` and `deploy.yml` entirely. A root `Dockerfile` builds the
app and `railway.toml` tells Railway to build/deploy from it; Railway's git
integration redeploys on every push to `main`, so there's no GitHub Actions
deploy step.

## Consequences

- Far less infrastructure to operate for a project at this stage: no
  Kubernetes, no container registry, no Bicep to maintain.
- Deploying to AKS again later is possible but not free — it would mean
  reintroducing Bicep/ACR/k8s manifests and a CI deploy step from scratch.
- Downstream: since compute leaves Azure, the Azure-native services that
  only made sense alongside AKS (Azure AI Search, Azure OpenAI, Azure Blob
  Storage) were reconsidered too — see ADR 0002, 0003, 0005.

# 0003 — OpenAI API directly, not Azure OpenAI

## Status

Accepted

## Context

`backend/config/azure_clients.py` authenticated both Azure OpenAI and
Azure AI Search against the enterprise tenant via one service principal.
The README's own architecture-decision list already stated "Model
provider: OpenAI API" as the intended default, inconsistent with the
Azure-OpenAI-specific code actually present — another leftover
inconsistency from the scaffold's drafting. Once Azure AI Search was
dropped (ADR 0002), keeping Azure OpenAI meant keeping the enterprise
service-principal auth machinery for one remaining Azure service.

## Decision

Use the OpenAI API directly (`OPENAI_API_KEY`), for both chat and
embeddings — `backend/config/llm_clients.py` replaces
`backend/config/azure_clients.py` entirely. The enterprise-tenant
service-principal auth pattern is removed, not kept dormant.

Kyma (kymaapi.com) was raised as a possible chat-model provider for
deployment specifically, but is **not** wired up: nothing in the codebase
depends on it yet, so building a provider-switch abstraction now would be
speculative. If it's picked up later, it happens in the same change that
actually stands up a Kyma account (see `agents.md`).

## Consequences

- No enterprise-tenant credentials (`AZURE_TENANT_ID`/`CLIENT_ID`/
  `CLIENT_SECRET`) anywhere in the stack.
- Local dev and production now use the identical LLM auth path — no more
  "this cloud service has no local equivalent" carve-out for OpenAI calls.
- If Kyma is adopted later for deployment specifically (chat only, per its
  own design — embeddings would likely stay on direct OpenAI), that's a
  deliberate follow-up decision, not something this ADR pre-approves.

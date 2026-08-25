"""Self-hosted Langfuse client for LLM/agent-specific tracing: prompts,
token usage, per-node graph execution, judge scores tied to a trace. Kept
separate from otel_setup.py because it captures LLM content, not just
service metrics — trace data stays on self-hosted infra rather than an
external SaaS.
"""


def configure_langfuse(host: str, public_key: str, secret_key: str):
    raise NotImplementedError

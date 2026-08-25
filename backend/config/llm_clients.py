"""Plain OpenAI client factories for chat and embeddings.

Kyma (kymaapi.com) is a possible future swap for the chat model on
deployment, but isn't wired up here — nothing in this codebase depends on
it yet, so there's no provider-switch abstraction to maintain. See
agents.md if that changes.
"""
import os
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@lru_cache
def get_chat_model(model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        model=model or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o"),
    )


@lru_cache
def get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

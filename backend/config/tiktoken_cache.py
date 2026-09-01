"""Import this module first, for its side effect, in anything that uses
tiktoken -- directly (backend/ingestion/chunking.py) or indirectly via
langchain_openai's embeddings client tokenizing text before sending it to
the API (backend/ingestion/embedding.py, backend/agents/tools/vector_search.py
through it).

tiktoken fetches its merge table from Azure Blob Storage over HTTPS on
first use if it isn't already in its local cache. That's fine in most
environments, but depending on outbound network access for a static,
versioned file on every cold process start is fragile regardless (and this
sandbox's Python doesn't trust the same CA store curl/the system do, the
same class of issue documented in build-log.md for next/font). Vendoring
the file (backend/ingestion/.tiktoken_cache/, hash-verified against
tiktoken's own expected SHA-256) and pointing TIKTOKEN_CACHE_DIR at it
sidesteps the fetch entirely, everywhere -- but only if this runs before
the *first* tiktoken call anywhere in the process, regardless of which
module makes it, hence a single shared import-for-side-effect module rather
than duplicating the setdefault per call site.
"""
import os
from pathlib import Path

os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "ingestion" / ".tiktoken_cache")
)

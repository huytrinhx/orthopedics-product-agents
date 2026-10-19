"""Token budgets for memory blocks (issue #7): count a block's tokens and
trim it to its cap, so every block memory adds to a prompt has a fixed,
measured cost instead of growing with the conversation.

Counts with tiktoken's cl100k_base, the same encoding ingestion/chunking.py
uses and the one vendored in ingestion/.tiktoken_cache (config/
tiktoken_cache.py points tiktoken at it, so counting never fetches anything).
The GPT-5.x models production runs use a newer encoding, so a count here is
close to what they bill, not exact -- the caps are starting points with
headroom, not limits a few tokens either way would break.
"""
from functools import cache

import tiktoken

from config import tiktoken_cache  # noqa: F401  (sets TIKTOKEN_CACHE_DIR)

_ELLIPSIS = "…"


@cache
def _encoding() -> tiktoken.Encoding:
    # Loaded on first use, not at import: agents/state.py imports this
    # module, and nothing that only needs the state schema should pay for
    # (or fail on) loading the merge table.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """`text` cut to at most `max_tokens` tokens, ending in an ellipsis when
    anything was cut. Decoding a token prefix can re-encode a token longer
    (a split multi-byte character), so it shrinks until it actually fits.
    """
    if max_tokens <= 0:
        return ""
    tokens = _encoding().encode(text)
    if len(tokens) <= max_tokens:
        return text
    keep = max_tokens - 1
    while keep > 0:
        truncated = _encoding().decode(tokens[:keep]).rstrip() + _ELLIPSIS
        if count_tokens(truncated) <= max_tokens:
            return truncated
        keep -= 1
    return ""


def fit_to_budget(header: str, entries: list[str], max_tokens: int) -> tuple[str, int]:
    """`header` plus as many of `entries` as fit in `max_tokens`, joined by
    blank lines, and its token count. Entries are oldest first, and the
    oldest go first when it's over. A single entry still over on its own is
    truncated, so the result never exceeds `max_tokens`. With no entries
    left, the block is empty (no header alone) and costs nothing.
    """
    kept = list(entries)
    while kept:
        block = "\n\n".join([header, *kept])
        tokens = count_tokens(block)
        if tokens <= max_tokens:
            return block, tokens
        if len(kept) == 1:
            block = truncate_to_tokens(block, max_tokens)
            return block, count_tokens(block)
        kept.pop(0)
    return "", 0

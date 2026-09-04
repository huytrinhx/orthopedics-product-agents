"""Extracts candidate single-word terms from free text for per-term graph
lookups (synonym resolution, Part matching) -- a query or a passage almost
never matches a Term/Part node as a whole string, so callers need to try
individual words, not the raw text.

Originally a private helper in backend/evals/compare_retrieval.py
(diagnostic-only); promoted here (ticket 21) so backend/agents/workflows/
deterministic.py's resolve_synonyms and resolve_skus nodes share the exact
same extraction logic and stopword list as that script, rather than each
maintaining their own drifting copy.
"""
# Common short words that spuriously substring-match real Part descriptions
# (e.g. "for" hits "SLEEVE, PARALLEL WIRES FOR PEG PLATE") -- excluded so
# candidate-term lookups aren't drowned in that noise. A diagnostic-tuned
# list, not a claim these words are never meaningful.
STOPWORDS = {
    "what", "whats", "which", "who", "how", "why", "when", "where",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "for", "of", "in", "on", "at", "to", "and", "or",
    "with", "this", "that", "these", "those", "it", "its", "i", "you",
    "set", "use", "used", "using",
}


def extract_candidate_terms(text: str) -> list[str]:
    """Lowercased, punctuation-stripped, deduped (order-preserving) words
    from `text`, dropping stopwords and anything 2 characters or shorter.
    """
    words = [w.strip(".,?!:;\"'()").lower() for w in text.split()]
    seen: set[str] = set()
    terms = []
    for word in words:
        if len(word) > 2 and word not in STOPWORDS and word not in seen:
            seen.add(word)
            terms.append(word)
    return terms


def with_singular_variants(terms: list[str]) -> list[str]:
    """Adds a trailing-s-stripped variant alongside each term that has one,
    deduped and order-preserving. Both the graph's Term nodes (synonym
    resolution) and Part descriptions are singular ("wire" -[:ALIAS_OF]->
    "guidepin"; "GUIDEPIN, MIS 3.5 PT...") while a real question almost
    always pluralizes ("the two 1.4mm wires") -- an exact-name or substring
    match against either never hits the plural as typed. Crude on purpose
    (no real stemmer): a plain trailing-s strip is a cheap substring/
    exact-match aid, not NLP, and both callers (resolve_synonyms,
    resolve_skus) need the exact same fix, not two drifting copies of it.
    """
    expanded = []
    for term in terms:
        expanded.append(term)
        if term.endswith("s") and len(term) > 3:
            expanded.append(term[:-1])
    return list(dict.fromkeys(expanded))

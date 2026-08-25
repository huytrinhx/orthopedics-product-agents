"""Splits ingested documents into retrieval-sized chunks."""


def chunk_document(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    raise NotImplementedError

"""Split text into overlapping whitespace-token windows."""
from __future__ import annotations


def chunk_text(text: str, size: int = 256, overlap: int = 32) -> list[str]:
    """Return windows of `size` whitespace tokens, each overlapping the previous by `overlap`.

    Windows start every `size - overlap` tokens and stop at the first window that reaches
    the last token, so no window is fully contained in the one before it.
    """
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than size")
    tokens = text.split()
    chunks: list[str] = []
    start = 0
    while tokens:
        if start + size >= len(tokens):
            chunks.append(" ".join(tokens[start:]))
            break
        chunks.append(" ".join(tokens[start:start + size]))
        start += step
    return chunks

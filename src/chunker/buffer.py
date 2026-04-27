"""Buffered sentence windows for embedding input."""

from __future__ import annotations

from chunker.models import SentenceSpan


def build_buffered_windows(
    sentences: list[SentenceSpan],
    buffer_size: int,
) -> list[str]:
    """Build embedding inputs with neighboring sentence context.

    For buffer_size=k, sentence i is represented by joining sentences
    [i-k, i+k] (clamped to document bounds), separated by a single space.
    Buffering affects embeddings only; final chunk text uses raw sentences.
    """
    if buffer_size < 0:
        raise ValueError("buffer_size must be >= 0")
    if not sentences:
        return []

    n = len(sentences)
    windows: list[str] = []
    for i in range(n):
        start = max(0, i - buffer_size)
        end = min(n, i + buffer_size + 1)
        windows.append(" ".join(s.text for s in sentences[start:end]))
    return windows

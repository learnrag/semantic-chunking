"""Chunk and diagnostics data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    start_sentence: int
    end_sentence: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_sentence": self.start_sentence,
            "end_sentence": self.end_sentence,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start_char: int
    end_char: int
    index: int


@dataclass(frozen=True)
class BoundaryDiagnostics:
    """Per-boundary diagnostics for inspect / evaluation."""

    similarities: list[float]
    distances: list[float]
    breakpoints: list[int]
    threshold: float
    threshold_method: str
    warnings: list[str] = field(default_factory=list)

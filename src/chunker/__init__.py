"""Semantic chunker: embedding-based topic boundary detection."""

from chunker.chunker import SemanticChunker
from chunker.embedder import Embedder, MockEmbedder
from chunker.models import BoundaryDiagnostics, Chunk, SentenceSpan

__all__ = [
    "BoundaryDiagnostics",
    "Chunk",
    "Embedder",
    "MockEmbedder",
    "SemanticChunker",
    "SentenceSpan",
]

"""Semantic chunker orchestration."""

from __future__ import annotations

from typing import Literal

from chunker.breakpoints import absolute_breakpoints, percentile_breakpoints
from chunker.buffer import build_buffered_windows
from chunker.constraints import (
    enforce_max_size,
    groups_from_breakpoints,
    merge_min_size,
)
from chunker.embedder import Embedder, l2_normalize
from chunker.models import BoundaryDiagnostics, Chunk
from chunker.sentence import RegexSentenceSegmenter, SentenceSegmenter
from chunker.similarity import adjacent_cosine_similarities, semantic_distances

ThresholdMethod = Literal["absolute", "percentile"]


class SemanticChunker:
    """Detect topic shifts via adjacent embedding distances and emit chunks."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold_method: ThresholdMethod = "percentile",
        threshold: float = 0.5,
        percentile: float = 95.0,
        buffer_size: int = 1,
        max_tokens: int = 500,
        min_tokens: int = 0,
        segmenter: SentenceSegmenter | None = None,
    ) -> None:
        if threshold_method not in ("absolute", "percentile"):
            raise ValueError("threshold_method must be 'absolute' or 'percentile'")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if min_tokens < 0:
            raise ValueError("min_tokens must be >= 0")
        if min_tokens > max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
        if buffer_size < 0:
            raise ValueError("buffer_size must be >= 0")

        self.embedder = embedder
        self.threshold_method = threshold_method
        self.threshold = threshold
        self.percentile = percentile
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.segmenter = segmenter or RegexSentenceSegmenter()

    def chunk(
        self,
        text: str,
        document_id: str = "doc",
    ) -> tuple[list[Chunk], BoundaryDiagnostics]:
        sentences = self.segmenter.split(text)
        if not sentences:
            empty = BoundaryDiagnostics(
                similarities=[],
                distances=[],
                breakpoints=[],
                threshold=0.0,
                threshold_method=self.threshold_method,
            )
            return [], empty

        if len(sentences) == 1:
            diagnostics = BoundaryDiagnostics(
                similarities=[],
                distances=[],
                breakpoints=[],
                threshold=0.0,
                threshold_method=self.threshold_method,
                warnings=["Single sentence document; no semantic boundaries possible."],
            )
            chunks = self._build_chunks_from_groups(
                groups_from_breakpoints(sentences, []),
                document_id,
                diagnostics,
            )
            return chunks, diagnostics

        windows = build_buffered_windows(sentences, self.buffer_size)
        embeddings = self.embedder.embed(windows)
        if not self.embedder.normalizes:
            embeddings = l2_normalize(embeddings)

        similarities = adjacent_cosine_similarities(
            embeddings, already_normalized=True
        )
        distances = semantic_distances(similarities)

        warn_msgs: list[str] = []
        if self.threshold_method == "absolute":
            breakpoints = absolute_breakpoints(distances, self.threshold)
            used_threshold = self.threshold
        else:
            breakpoints, used_threshold, warn_msgs = percentile_breakpoints(
                distances,
                self.percentile,
                n_sentences=len(sentences),
            )

        diagnostics = BoundaryDiagnostics(
            similarities=[float(s) for s in similarities],
            distances=[float(d) for d in distances],
            breakpoints=list(breakpoints),
            threshold=float(used_threshold),
            threshold_method=self.threshold_method,
            warnings=warn_msgs,
        )

        groups = groups_from_breakpoints(sentences, breakpoints)
        groups = enforce_max_size(groups, sentences, self.max_tokens)
        if self.min_tokens > 0:
            groups = merge_min_size(
                groups, sentences, self.min_tokens, self.max_tokens
            )

        chunks = self._build_chunks_from_groups(groups, document_id, diagnostics)
        return chunks, diagnostics

    def _build_chunks_from_groups(
        self,
        groups: list,
        document_id: str,
        diagnostics: BoundaryDiagnostics,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, group in enumerate(groups):
            metadata = {
                "strategy": "semantic",
                "threshold_method": diagnostics.threshold_method,
                "threshold": diagnostics.threshold,
                "max_tokens": self.max_tokens,
                "min_tokens": self.min_tokens,
                "buffer_size": self.buffer_size,
                "embedding_model": self.embedder.model_name,
                "embedding_dim": self.embedder.dimension,
                "normalization": (
                    "l2" if self.embedder.normalizes else "l2-post"
                ),
            }
            if group.oversized:
                metadata["oversized"] = True
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}_chunk_{index:04d}",
                    document_id=document_id,
                    chunk_index=index,
                    text=group.text,
                    start_char=group.start_char,
                    end_char=group.end_char,
                    start_sentence=group.start_sentence,
                    end_sentence=group.end_sentence,
                    metadata=metadata,
                )
            )
        return chunks

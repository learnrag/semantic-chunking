"""Embedding interface and deterministic mock embedder."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    @property
    def normalizes(self) -> bool:
        ...


def validate_embeddings(matrix: np.ndarray, expected_count: int) -> np.ndarray:
    """Ensure embedding matrix is finite and correctly shaped."""
    if expected_count == 0:
        if matrix.size != 0 and matrix.ndim == 2 and matrix.shape[0] != 0:
            raise ValueError("Expected empty embedding matrix for zero texts")
        return matrix.reshape(0, matrix.shape[1] if matrix.ndim == 2 and matrix.shape[1] else 0)

    if matrix.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] != expected_count:
        raise ValueError(
            f"Expected {expected_count} embeddings, got {matrix.shape[0]}"
        )
    if matrix.shape[1] == 0:
        raise ValueError("Embedding dimension must be > 0")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or Inf values")
    return matrix


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


class MockEmbedder:
    """Deterministic hash-based embedder for tests (no model download)."""

    def __init__(self, dimension: int = 32, seed: int = 0) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self._dimension = dimension
        self._seed = seed

    @property
    def model_name(self) -> str:
        return f"mock-embedder-d{self._dimension}-s{self._seed}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalizes(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float64)
        rows = [self._vector_for(text) for text in texts]
        matrix = np.stack(rows, axis=0)
        return validate_embeddings(l2_normalize(matrix), len(texts))

    def _vector_for(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(f"{self._seed}:{text}".encode("utf-8")).digest()
        # Expand digest into floats in [-1, 1].
        values = np.frombuffer(digest * ((self._dimension // 32) + 1), dtype=np.uint8)
        values = values[: self._dimension].astype(np.float64)
        return (values / 127.5) - 1.0


class SentenceTransformerEmbedder:
    """Local sentence-transformers wrapper (model name is injectable)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        normalize: bool = True,
        revision: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs: dict = {}
        if revision is not None:
            kwargs["revision"] = revision
        self._model = SentenceTransformer(model_name, **kwargs)
        self._model_name = model_name
        self._revision = revision
        self._normalize = normalize
        # Probe dimension with a tiny encode.
        probe = self._model.encode(["dimension probe"], normalize_embeddings=False)
        self._dimension = int(np.asarray(probe).shape[-1])

    @property
    def model_name(self) -> str:
        if self._revision:
            return f"{self._model_name}@{self._revision}"
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalizes(self) -> bool:
        return self._normalize

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float64)
        matrix = np.asarray(
            self._model.encode(
                texts,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )
        return validate_embeddings(matrix, len(texts))

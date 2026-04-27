"""Adjacent cosine similarity and semantic distance."""

from __future__ import annotations

import numpy as np

from chunker.embedder import l2_normalize, validate_embeddings


def adjacent_cosine_similarities(
    embeddings: np.ndarray,
    *,
    already_normalized: bool = False,
) -> np.ndarray:
    """Return cosine similarity between each row and the next.

    Result length is n-1 for n embeddings. Empty or single-row input
    yields an empty array.
    """
    matrix = validate_embeddings(embeddings, embeddings.shape[0] if embeddings.ndim == 2 else 0)
    if matrix.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)

    if not already_normalized:
        matrix = l2_normalize(matrix)

    # After L2 norm, cosine == dot product.
    sims = np.sum(matrix[:-1] * matrix[1:], axis=1)
    # Numerical clamp into [-1, 1].
    return np.clip(sims, -1.0, 1.0)


def semantic_distances(similarities: np.ndarray) -> np.ndarray:
    """Convert cosine similarities to semantic distances (1 - sim)."""
    return 1.0 - np.asarray(similarities, dtype=np.float64)

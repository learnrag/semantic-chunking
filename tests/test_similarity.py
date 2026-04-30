"""Tests for embeddings and similarity."""

import numpy as np
import pytest

from chunker.embedder import MockEmbedder, validate_embeddings
from chunker.similarity import adjacent_cosine_similarities, semantic_distances


def test_mock_embedder_deterministic():
    emb = MockEmbedder(dimension=16, seed=1)
    a = emb.embed(["alpha", "beta"])
    b = emb.embed(["alpha", "beta"])
    assert a.shape == (2, 16)
    np.testing.assert_allclose(a, b)
    norms = np.linalg.norm(a, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)


def test_validate_rejects_nan():
    bad = np.array([[1.0, np.nan], [0.0, 1.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_embeddings(bad, 2)


def test_adjacent_cosine_and_distance():
    # Identical unit vectors -> sim 1, distance 0
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sims = adjacent_cosine_similarities(matrix, already_normalized=True)
    np.testing.assert_allclose(sims, [1.0, 0.0], atol=1e-6)
    dists = semantic_distances(sims)
    np.testing.assert_allclose(dists, [0.0, 1.0], atol=1e-6)


def test_adjacent_empty():
    sims = adjacent_cosine_similarities(np.zeros((1, 4)), already_normalized=True)
    assert sims.shape == (0,)

"""Tests for evaluation metrics."""

from chunker.evaluate import boundary_scores, chunk_size_stats, load_gold_dataset
from chunker.models import Chunk
from pathlib import Path


def test_boundary_scores():
    scores = boundary_scores([1, 3, 5], [3, 5, 7])
    assert scores.true_positives == 2
    assert scores.precision == 2 / 3
    assert scores.recall == 2 / 3


def test_boundary_scores_empty():
    scores = boundary_scores([], [])
    assert scores.f1 == 0.0


def test_chunk_size_stats():
    chunks = [
        Chunk("a", "d", 0, "one two", 0, 7, 0, 0, {}),
        Chunk("b", "d", 1, "three four five", 8, 23, 1, 1, {}),
    ]
    stats = chunk_size_stats(chunks)
    assert stats.chunk_count == 2
    assert stats.mean_tokens == 2.5


def test_load_gold():
    path = Path(__file__).parent / "fixtures" / "gold.json"
    docs = load_gold_dataset(path)
    assert len(docs) >= 4
    assert "gold_boundaries" in docs[0]

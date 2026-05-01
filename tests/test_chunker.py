"""Tests for SemanticChunker orchestration."""

from chunker.chunker import SemanticChunker
from chunker.embedder import MockEmbedder


def test_chunk_empty():
    chunker = SemanticChunker(MockEmbedder(), threshold_method="absolute", threshold=0.5)
    chunks, diag = chunker.chunk("")
    assert chunks == []
    assert diag.breakpoints == []


def test_chunk_single_sentence():
    chunker = SemanticChunker(MockEmbedder(), max_tokens=100)
    chunks, diag = chunker.chunk("Only one sentence here.")
    assert len(chunks) == 1
    assert chunks[0].start_sentence == 0
    assert diag.warnings


def test_absolute_threshold_splits():
    # MockEmbedder is hash-based; use absolute with low threshold to force many breaks,
    # and high threshold for few.
    text = (
        "Alpha topic stays here. Beta continues the idea. "
        "Gamma shifts somewhere. Delta ends the note."
    )
    low = SemanticChunker(
        MockEmbedder(seed=0),
        threshold_method="absolute",
        threshold=0.01,
        max_tokens=500,
        buffer_size=0,
    )
    high = SemanticChunker(
        MockEmbedder(seed=0),
        threshold_method="absolute",
        threshold=1.5,
        max_tokens=500,
        buffer_size=0,
    )
    low_chunks, low_diag = low.chunk(text, "doc")
    high_chunks, high_diag = high.chunk(text, "doc")
    assert len(low_diag.breakpoints) >= len(high_diag.breakpoints)
    assert high_diag.breakpoints == []
    assert len(high_chunks) == 1
    assert all(c.metadata["strategy"] == "semantic" for c in low_chunks)


def test_percentile_mode_runs():
    text = "One. Two. Three. Four. Five. Six."
    chunker = SemanticChunker(
        MockEmbedder(),
        threshold_method="percentile",
        percentile=90,
        buffer_size=1,
        max_tokens=50,
    )
    chunks, diag = chunker.chunk(text)
    assert chunks
    assert diag.threshold_method == "percentile"
    assert 0 <= diag.threshold <= 2


def test_max_tokens_enforced():
    text = "Word one. Word two. Word three. Word four. Word five. Word six."
    chunker = SemanticChunker(
        MockEmbedder(),
        threshold_method="absolute",
        threshold=2.0,  # no semantic breaks
        max_tokens=4,
        buffer_size=0,
    )
    chunks, _ = chunker.chunk(text)
    assert len(chunks) > 1
    for c in chunks:
        if not c.metadata.get("oversized"):
            assert len(c.text.split()) <= 4

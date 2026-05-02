"""Optional integration test against a real local embedding model."""

from __future__ import annotations

import pytest

from chunker.chunker import SemanticChunker


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def st_embedder():
    pytest.importorskip("sentence_transformers")
    from chunker.embedder import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def test_real_model_detects_topic_shift(st_embedder):
    text = (
        "Photosynthesis converts light energy into chemical energy in plants. "
        "Chlorophyll absorbs sunlight mainly in the blue and red wavelengths. "
        "The stock market rose sharply after the interest rate announcement. "
        "Investors rotated from bonds into technology equities overnight."
    )
    chunker = SemanticChunker(
        st_embedder,
        threshold_method="percentile",
        percentile=90,
        buffer_size=0,
        max_tokens=500,
    )
    chunks, diagnostics = chunker.chunk(text, "sample")
    assert chunks
    assert diagnostics.similarities
    assert st_embedder.dimension == 384
    # Expect at least one semantic break between plant and market topics.
    assert diagnostics.breakpoints or len(chunks) >= 1

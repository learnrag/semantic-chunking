"""Tests for breakpoint detection and size constraints."""

import numpy as np
import pytest

from chunker.breakpoints import absolute_breakpoints, percentile_breakpoints
from chunker.constraints import (
    enforce_max_size,
    groups_from_breakpoints,
    merge_min_size,
    token_count,
)
from chunker.models import SentenceSpan


def _sents(*texts: str) -> list[SentenceSpan]:
    spans = []
    offset = 0
    for i, t in enumerate(texts):
        spans.append(SentenceSpan(text=t, start_char=offset, end_char=offset + len(t), index=i))
        offset += len(t) + 1
    return spans


def test_absolute_breakpoints():
    dists = np.array([0.1, 0.8, 0.2])
    assert absolute_breakpoints(dists, 0.5) == [1]


def test_percentile_breakpoints():
    dists = np.array([0.1, 0.2, 0.3, 0.9, 0.15])
    bps, thr, warns = percentile_breakpoints(dists, 80, n_sentences=6)
    assert thr == pytest.approx(float(np.percentile(dists, 80)))
    assert bps
    assert all(dists[i] >= thr for i in bps)
    assert warns == []


def test_percentile_short_doc_warns():
    dists = np.array([0.5])
    with pytest.warns(UserWarning, match="unreliable"):
        bps, thr, warns = percentile_breakpoints(dists, 95, n_sentences=2)
    assert warns


def test_max_min_constraints():
    sentences = _sents(
        "one two three",
        "four five six",
        "seven eight nine ten",
        "eleven",
    )
    groups = groups_from_breakpoints(sentences, [1])
    assert len(groups) == 2

    # Force max split: tiny max_tokens
    split = enforce_max_size(groups, sentences, max_tokens=3)
    assert all(g.tokens <= 3 or g.oversized for g in split)

    # Oversized single sentence
    long = _sents("a b c d e f g h i j")
    groups = groups_from_breakpoints(long, [])
    out = enforce_max_size(groups, long, max_tokens=3)
    assert len(out) == 1
    assert out[0].oversized is True

    # Merge min
    tiny = _sents("a", "b", "c d e f")
    groups = groups_from_breakpoints(tiny, [0, 1])
    merged = merge_min_size(groups, tiny, min_tokens=2, max_tokens=10)
    assert len(merged) < len(groups)
    assert token_count("a") == 1

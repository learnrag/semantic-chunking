"""Breakpoint detection from semantic distances."""

from __future__ import annotations

import warnings

import numpy as np


def absolute_breakpoints(distances: np.ndarray, threshold: float) -> list[int]:
    """Return indices i where a boundary is placed after sentence i."""
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    dists = np.asarray(distances, dtype=np.float64)
    return [i for i, d in enumerate(dists) if d > threshold]


def percentile_breakpoints(
    distances: np.ndarray,
    percentile: float,
    *,
    n_sentences: int | None = None,
) -> tuple[list[int], float, list[str]]:
    """Adaptive breakpoints using a high percentile of distances.

    Returns (breakpoint_indices, computed_threshold, warnings).
    """
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")

    dists = np.asarray(distances, dtype=np.float64)
    warn_msgs: list[str] = []
    sentence_count = n_sentences if n_sentences is not None else (
        len(dists) + 1 if len(dists) else 0
    )

    if sentence_count < 5:
        msg = (
            f"Percentile threshold is unreliable with only {sentence_count} "
            "sentences; prefer absolute threshold for short documents."
        )
        warn_msgs.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    if dists.size == 0:
        return [], 0.0, warn_msgs

    threshold = float(np.percentile(dists, percentile))
    breakpoints = [i for i, d in enumerate(dists) if d >= threshold]
    return breakpoints, threshold, warn_msgs

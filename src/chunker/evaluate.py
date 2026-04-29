"""Boundary evaluation metrics against gold labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chunker.constraints import token_count
from chunker.models import Chunk


@dataclass(frozen=True)
class BoundaryScores:
    precision: float
    recall: float
    f1: float
    predicted: int
    gold: int
    true_positives: int


@dataclass(frozen=True)
class ChunkSizeStats:
    chunk_count: int
    mean_tokens: float
    p95_tokens: float


def boundary_scores(
    predicted: list[int],
    gold: list[int],
) -> BoundaryScores:
    pred_set = set(predicted)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return BoundaryScores(
        precision=precision,
        recall=recall,
        f1=f1,
        predicted=len(pred_set),
        gold=len(gold_set),
        true_positives=tp,
    )


def chunk_size_stats(chunks: list[Chunk]) -> ChunkSizeStats:
    if not chunks:
        return ChunkSizeStats(chunk_count=0, mean_tokens=0.0, p95_tokens=0.0)
    sizes = np.array([token_count(c.text) for c in chunks], dtype=np.float64)
    return ChunkSizeStats(
        chunk_count=len(chunks),
        mean_tokens=float(sizes.mean()),
        p95_tokens=float(np.percentile(sizes, 95)),
    )


def load_gold_dataset(path: Path) -> list[dict[str, Any]]:
    """Load gold JSON: a list of docs or a single doc object."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        raise ValueError("Gold dataset must be a JSON object or array")
    return data


def predicted_boundaries_from_chunks(chunks: list[Chunk]) -> list[int]:
    """Boundary after end_sentence of each chunk except the last."""
    if len(chunks) <= 1:
        return []
    return [c.end_sentence for c in chunks[:-1]]

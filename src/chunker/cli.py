"""CLI for chunk, inspect, benchmark, and compare."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chunker.baseline import fixed_token_chunk, paragraph_chunk
from chunker.chunker import SemanticChunker
from chunker.embedder import MockEmbedder
from chunker.evaluate import (
    boundary_scores,
    chunk_size_stats,
    load_gold_dataset,
    predicted_boundaries_from_chunks,
)
from chunker.io import load_text, write_jsonl
from chunker.sentence import RegexSentenceSegmenter


def _document_id(path: Path, override: str | None) -> str:
    return override if override is not None else path.stem


def _resolve_embedder(args: argparse.Namespace):
    if getattr(args, "mock", False):
        return MockEmbedder(dimension=getattr(args, "mock_dim", 32))
    from chunker.embedder import SentenceTransformerEmbedder

    model = getattr(args, "model", None) or "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformerEmbedder(model_name=model)


def _add_chunker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local sentence-transformers model name",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic MockEmbedder (no model download)",
    )
    parser.add_argument("--mock-dim", type=int, default=32, help="Mock embedder dim")
    parser.add_argument("--buffer-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--document-id", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Percentile breakpoint threshold (default mode if set or neither given)",
    )
    group.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute semantic distance threshold",
    )


def _build_chunker(args: argparse.Namespace) -> SemanticChunker:
    embedder = _resolve_embedder(args)
    if args.threshold is not None:
        method = "absolute"
        threshold = args.threshold
        percentile = 95.0
    else:
        method = "percentile"
        threshold = 0.5
        percentile = args.percentile if args.percentile is not None else 95.0
    return SemanticChunker(
        embedder,
        threshold_method=method,
        threshold=threshold,
        percentile=percentile,
        buffer_size=args.buffer_size,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )


def cmd_chunk(args: argparse.Namespace) -> int:
    text = load_text(args.path)
    chunker = _build_chunker(args)
    chunks, _ = chunker.chunk(text, _document_id(args.path, args.document_id))
    write_jsonl(args.output, chunks)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    text = load_text(args.path)
    chunker = _build_chunker(args)
    sentences = chunker.segmenter.split(text)
    _, diagnostics = chunker.chunk(text, _document_id(args.path, args.document_id))

    break_set = set(diagnostics.breakpoints)
    print(
        f"threshold_method={diagnostics.threshold_method} "
        f"threshold={diagnostics.threshold:.4f} "
        f"sentences={len(sentences)} "
        f"breakpoints={diagnostics.breakpoints}"
    )
    for w in diagnostics.warnings:
        print(f"warning: {w}")

    if args.show_similarity:
        for i, sent in enumerate(sentences):
            preview = sent.text.replace("\n", " ")
            if len(preview) > 72:
                preview = preview[:69] + "..."
            if i < len(diagnostics.similarities):
                sim = diagnostics.similarities[i]
                marker = "  ← BREAK" if i in break_set else ""
                print(f"Sentence {i + 1} → next similarity {sim:.2f}{marker}")
                print(f"  {preview}")
            else:
                print(f"Sentence {i + 1} → (end)")
                print(f"  {preview}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    docs = load_gold_dataset(args.gold)
    chunker = _build_chunker(args)
    segmenter = RegexSentenceSegmenter()

    all_pred: list[int] = []
    all_gold: list[int] = []
    # Per-doc scores averaged; also micro-average via concatenation with offsets.
    # Use per-doc evaluation then macro-average F1 components.
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    total_chunks = 0
    all_sizes: list[float] = []

    for doc in docs:
        doc_id = doc["id"]
        sentences = doc["sentences"]
        gold = list(doc["gold_boundaries"])
        text = " ".join(sentences)
        # Prefer provided sentences via a trivial segmenter override when counts match.
        chunks, diagnostics = chunker.chunk(text, doc_id)
        # Align predicted boundaries to gold sentence indices when possible.
        pred = list(diagnostics.breakpoints)
        # If size constraints added/removed boundaries, use chunk ends.
        size_pred = predicted_boundaries_from_chunks(chunks)
        # Prefer semantic breakpoints for boundary metrics (PRD).
        scores = boundary_scores(pred, gold)
        precisions.append(scores.precision)
        recalls.append(scores.recall)
        f1s.append(scores.f1)
        all_pred.extend(pred)
        all_gold.extend(gold)
        stats = chunk_size_stats(chunks)
        total_chunks += stats.chunk_count
        all_sizes.extend([len(c.text.split()) for c in chunks])
        print(
            f"{doc_id}: P={scores.precision:.3f} R={scores.recall:.3f} "
            f"F1={scores.f1:.3f} chunks={stats.chunk_count} "
            f"semantic_breaks={pred} gold={gold} size_breaks={size_pred}"
        )
        _ = segmenter  # reserved for future gold-sentence alignment

    n = max(len(docs), 1)
    macro_p = sum(precisions) / n
    macro_r = sum(recalls) / n
    macro_f1 = sum(f1s) / n
    micro = boundary_scores(all_pred, all_gold)
    mean_size = sum(all_sizes) / len(all_sizes) if all_sizes else 0.0
    import numpy as np

    p95 = float(np.percentile(all_sizes, 95)) if all_sizes else 0.0
    print("---")
    print(
        f"macro: P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f} | "
        f"micro: P={micro.precision:.3f} R={micro.recall:.3f} F1={micro.f1:.3f}"
    )
    print(
        f"chunks={total_chunks} mean_tokens={mean_size:.1f} p95_tokens={p95:.1f}"
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    text = load_text(args.path)
    doc_id = _document_id(args.path, args.document_id)
    embedder = _resolve_embedder(args)

    abs_chunker = SemanticChunker(
        embedder,
        threshold_method="absolute",
        threshold=args.threshold,
        buffer_size=args.buffer_size,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    pct_chunker = SemanticChunker(
        embedder,
        threshold_method="percentile",
        percentile=args.percentile,
        buffer_size=args.buffer_size,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )

    abs_chunks, abs_diag = abs_chunker.chunk(text, doc_id)
    pct_chunks, pct_diag = pct_chunker.chunk(text, doc_id)
    para = paragraph_chunk(text, doc_id)
    fixed = fixed_token_chunk(text, args.max_tokens, doc_id)

    def summarize(name: str, chunks, breaks=None, threshold=None):
        stats = chunk_size_stats(chunks)
        extra = ""
        if breaks is not None:
            extra = f" breaks={breaks}"
        if threshold is not None:
            extra += f" threshold={threshold:.4f}"
        print(
            f"{name}: chunks={stats.chunk_count} "
            f"mean_tokens={stats.mean_tokens:.1f} "
            f"p95_tokens={stats.p95_tokens:.1f}{extra}"
        )

    summarize("semantic_absolute", abs_chunks, abs_diag.breakpoints, abs_diag.threshold)
    summarize("semantic_percentile", pct_chunks, pct_diag.breakpoints, pct_diag.threshold)
    summarize("paragraph", para)
    summarize("fixed_token", fixed)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chunker", description="Semantic text chunker")
    sub = parser.add_subparsers(dest="command", required=True)

    p_chunk = sub.add_parser("chunk", help="Chunk a document to JSONL")
    p_chunk.add_argument("path", type=Path)
    p_chunk.add_argument("-o", "--output", type=Path, default=None)
    _add_chunker_args(p_chunk)
    p_chunk.set_defaults(func=cmd_chunk)

    p_inspect = sub.add_parser("inspect", help="Show similarity diagnostics")
    p_inspect.add_argument("path", type=Path)
    p_inspect.add_argument("--show-similarity", action="store_true", default=True)
    _add_chunker_args(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    p_bench = sub.add_parser("benchmark", help="Evaluate against gold boundaries")
    p_bench.add_argument("gold", type=Path, help="Path to gold JSON dataset")
    _add_chunker_args(p_bench)
    p_bench.set_defaults(func=cmd_benchmark)

    p_compare = sub.add_parser("compare", help="Compare strategies side by side")
    p_compare.add_argument("path", type=Path)
    p_compare.add_argument("--threshold", type=float, default=0.20)
    p_compare.add_argument("--percentile", type=float, default=95.0)
    p_compare.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_compare.add_argument("--mock", action="store_true")
    p_compare.add_argument("--mock-dim", type=int, default=32)
    p_compare.add_argument("--buffer-size", type=int, default=1)
    p_compare.add_argument("--max-tokens", type=int, default=500)
    p_compare.add_argument("--min-tokens", type=int, default=0)
    p_compare.add_argument("--document-id", default=None)
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

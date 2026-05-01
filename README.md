# Semantic Chunker

Local, embedding-based semantic text chunker for learning RAG pipelines.

It does **not** ask an LLM where to split. It embeds sentences (optionally with a
neighbor buffer), measures adjacent cosine distance, and places boundaries where
distance exceeds an absolute or percentile threshold — then enforces min/max
token size.

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # core + tests (MockEmbedder)
pip install -e ".[embed]"        # + sentence-transformers for real models
```

Default local model: `sentence-transformers/all-MiniLM-L6-v2`.

## CLI

```bash
# Chunk to JSONL (mock embedder — no download)
chunker chunk data/sample.txt --mock --percentile 95 --max-tokens 500 -o out.jsonl

# Real local model
chunker chunk data/sample.txt --model sentence-transformers/all-MiniLM-L6-v2 --percentile 95

# Inspect similarities and breaks
chunker inspect data/sample.txt --mock --show-similarity --threshold 0.25

# Gold-boundary benchmark
chunker benchmark tests/fixtures/gold.json --mock --percentile 90

# Compare absolute vs percentile vs baselines
chunker compare data/sample.txt --mock --threshold 0.20 --percentile 95
```

## HTTP API

```bash
chunker-api
# or: uvicorn chunker.api:app --host 0.0.0.0 --port 8000
```

`POST /chunk` with JSON body (`text`, `mock`, `percentile` / `threshold`, …).
`GET /health`.

## Docker

```bash
docker compose up --build
```

Image installs `.[embed]` and serves the API on port 8000 (`PORT` env supported).

## Algorithm

1. Split text into sentences (regex segmenter; injectable).
2. Build buffered windows: for `buffer_size=k`, sentence `i` embeds as
   `join(S[i-k : i+k+1])` (edge-clamped). Buffering affects embeddings only.
3. Embed → L2-normalize if needed → adjacent cosine similarity → distance `1 - sim`.
4. Breakpoints:
   - **absolute**: `distance_i > threshold`
   - **percentile**: `threshold = percentile(distances, p)`; split when `distance_i >= threshold`
5. Enforce `max_tokens` (whitespace tokens); merge undersized chunks when `min_tokens > 0`.
6. Emit chunks + diagnostics (similarities, distances, breakpoints, model metadata).

### Failure cases

| Case | Behavior |
|------|----------|
| < 2 sentences | Single chunk; no distances |
| Short docs + percentile | Warning; prefer absolute |
| Flat distances | Few/no breaks |
| Single sentence > max_tokens | Kept; `metadata.oversized=true` |
| Bad embeddings (NaN/Inf) | Hard error |
| Missing sentence-transformers | Clear ImportError; use `--mock` |

## Library usage

```python
from chunker import SemanticChunker, MockEmbedder

chunker = SemanticChunker(
    MockEmbedder(),
    threshold_method="percentile",
    percentile=95,
    buffer_size=1,
    max_tokens=500,
)
chunks, diagnostics = chunker.chunk(text, document_id="doc1")
```

## Tests

```bash
pytest                 # unit + CLI/API with MockEmbedder
pytest -m integration  # downloads MiniLM (needs [embed])
```

## Gold dataset

See `tests/fixtures/gold.json`: abrupt shifts, gradual prose, short docs,
all-similar sentences, and list-like noise with labelled `gold_boundaries`
(boundary after sentence index `i`).

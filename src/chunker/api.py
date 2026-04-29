"""FastAPI HTTP API for semantic chunking."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chunker.chunker import SemanticChunker
from chunker.embedder import MockEmbedder

ThresholdMethod = Literal["absolute", "percentile"]

app = FastAPI(
    title="semantic-chunker",
    description="Embedding-based semantic text chunker",
    version="0.1.0",
)

# Lazy singleton for the default local model.
_default_embedder = None


def _get_default_embedder(model_name: str):
    global _default_embedder
    if _default_embedder is not None and _default_embedder.model_name.startswith(
        model_name
    ):
        return _default_embedder
    from chunker.embedder import SentenceTransformerEmbedder

    _default_embedder = SentenceTransformerEmbedder(model_name=model_name)
    return _default_embedder


class ChunkRequest(BaseModel):
    text: str
    document_id: str = "doc"
    threshold_method: ThresholdMethod = "percentile"
    threshold: float = Field(0.5, ge=0)
    percentile: float = Field(95.0, ge=0, le=100)
    buffer_size: int = Field(1, ge=0)
    max_tokens: int = Field(500, ge=1)
    min_tokens: int = Field(0, ge=0)
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    mock: bool = False


class ChunkResponse(BaseModel):
    chunks: list[dict[str, Any]]
    count: int
    diagnostics: dict[str, Any]


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _build_chunker(req: ChunkRequest) -> SemanticChunker:
    try:
        if req.mock:
            embedder = MockEmbedder()
        else:
            embedder = _get_default_embedder(req.model)
        return SemanticChunker(
            embedder,
            threshold_method=req.threshold_method,
            threshold=req.threshold,
            percentile=req.percentile,
            buffer_size=req.buffer_size,
            max_tokens=req.max_tokens,
            min_tokens=req.min_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/chunk", response_model=ChunkResponse)
def chunk_endpoint(req: ChunkRequest) -> ChunkResponse:
    chunker = _build_chunker(req)
    try:
        chunks, diagnostics = chunker.chunk(req.text, req.document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ChunkResponse(
        chunks=[c.to_dict() for c in chunks],
        count=len(chunks),
        diagnostics={
            "similarities": diagnostics.similarities,
            "distances": diagnostics.distances,
            "breakpoints": diagnostics.breakpoints,
            "threshold": diagnostics.threshold,
            "threshold_method": diagnostics.threshold_method,
            "warnings": diagnostics.warnings,
        },
    )


def run() -> None:
    import uvicorn

    uvicorn.run("chunker.api:app", host="0.0.0.0", port=8000, reload=False)

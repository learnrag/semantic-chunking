"""Non-semantic baselines for compare command."""

from __future__ import annotations

from chunker.constraints import token_count
from chunker.models import Chunk
from chunker.sentence import RegexSentenceSegmenter


def paragraph_chunk(text: str, document_id: str = "doc") -> list[Chunk]:
    """Split on blank lines (paragraph baseline)."""
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    offset = 0
    parts = text.split("\n\n")
    for part in parts:
        # Account for separator consumed by split except trailing.
        raw = part
        start = text.find(raw, offset)
        if start < 0:
            start = offset
        end = start + len(raw)
        offset = end + 2  # skip \n\n
        stripped = raw.strip()
        if not stripped:
            continue
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        core_start = start + lead
        core_end = end - trail
        core = text[core_start:core_end]
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}_chunk_{idx:04d}",
                document_id=document_id,
                chunk_index=idx,
                text=core,
                start_char=core_start,
                end_char=core_end,
                start_sentence=-1,
                end_sentence=-1,
                metadata={"strategy": "paragraph"},
            )
        )
    return chunks


def fixed_token_chunk(
    text: str,
    max_tokens: int,
    document_id: str = "doc",
) -> list[Chunk]:
    """Greedy sentence packing up to max_tokens (fixed-size baseline)."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    sentences = RegexSentenceSegmenter().split(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    start = 0
    tokens = 0
    for i, sent in enumerate(sentences):
        st = token_count(sent.text)
        if i > start and tokens + st > max_tokens:
            chunks.append(_pack(sentences, start, i - 1, document_id, len(chunks)))
            start = i
            tokens = st
        else:
            tokens += st
    chunks.append(_pack(sentences, start, len(sentences) - 1, document_id, len(chunks)))
    return chunks


def _pack(sentences, start, end, document_id, index) -> Chunk:
    text = " ".join(s.text for s in sentences[start : end + 1])
    return Chunk(
        chunk_id=f"{document_id}_chunk_{index:04d}",
        document_id=document_id,
        chunk_index=index,
        text=text,
        start_char=sentences[start].start_char,
        end_char=sentences[end].end_char,
        start_sentence=start,
        end_sentence=end,
        metadata={"strategy": "fixed_token"},
    )

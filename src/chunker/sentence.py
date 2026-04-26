"""Sentence segmentation with character offset provenance."""

from __future__ import annotations

import re
from typing import Protocol

from chunker.models import SentenceSpan

# Split after . ! ? (optionally followed by quotes/brackets) when whitespace follows.
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])(?:['\"”)\]\}]*)(?=\s+)|(?<=[.!?])(?:['\"”)\]\}]*)$"
)


class SentenceSegmenter(Protocol):
    def split(self, text: str) -> list[SentenceSpan]:
        ...


class RegexSentenceSegmenter:
    """Lightweight regex sentence splitter that preserves character offsets."""

    def split(self, text: str) -> list[SentenceSpan]:
        if not text:
            return []
        if not text.strip():
            return [
                SentenceSpan(text=text, start_char=0, end_char=len(text), index=0)
            ]

        parts: list[str] = []
        last = 0
        for match in _SENTENCE_RE.finditer(text):
            end = match.end()
            parts.append(text[last:end])
            last = end
        if last < len(text):
            parts.append(text[last:])

        spans: list[SentenceSpan] = []
        offset = 0
        for part in parts:
            # Preserve leading whitespace on the first piece; strip only
            # leading whitespace from subsequent pieces for clean boundaries.
            if spans:
                stripped = part.lstrip()
                lead = len(part) - len(stripped)
                offset += lead
                part = stripped
            if not part:
                continue
            # Trim trailing whitespace from the span text but keep end_char
            # pointing past any trailing whitespace belonging to this sentence.
            trailing = len(part) - len(part.rstrip())
            core = part if trailing == 0 else part[:-trailing]
            if not core:
                offset += len(part)
                continue
            start = offset
            end = offset + len(core)
            spans.append(
                SentenceSpan(text=core, start_char=start, end_char=end, index=len(spans))
            )
            offset += len(part)

        if not spans and text.strip():
            core = text.strip()
            start = text.index(core)
            spans.append(
                SentenceSpan(
                    text=core,
                    start_char=start,
                    end_char=start + len(core),
                    index=0,
                )
            )
        return spans

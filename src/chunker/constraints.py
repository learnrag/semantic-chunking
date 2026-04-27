"""Whitespace token counting and min/max chunk size constraints."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\S+")


def token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


@dataclass(frozen=True)
class SentenceGroup:
    """Contiguous sentence range forming a candidate chunk."""

    start_sentence: int
    end_sentence: int  # inclusive
    text: str
    start_char: int
    end_char: int
    oversized: bool = False

    @property
    def tokens(self) -> int:
        return token_count(self.text)


def groups_from_breakpoints(
    sentences: list,  # SentenceSpan
    breakpoints: list[int],
) -> list[SentenceGroup]:
    """Split sentences into groups at breakpoint indices (boundary after i)."""
    if not sentences:
        return []

    breaks = sorted(set(breakpoints))
    groups: list[SentenceGroup] = []
    start = 0
    for bp in breaks:
        if bp < start or bp >= len(sentences) - 1:
            continue
        end = bp
        groups.append(_group(sentences, start, end))
        start = end + 1
    groups.append(_group(sentences, start, len(sentences) - 1))
    return groups


def _group(sentences: list, start: int, end: int) -> SentenceGroup:
    text = " ".join(s.text for s in sentences[start : end + 1])
    return SentenceGroup(
        start_sentence=start,
        end_sentence=end,
        text=text,
        start_char=sentences[start].start_char,
        end_char=sentences[end].end_char,
    )


def enforce_max_size(
    groups: list[SentenceGroup],
    sentences: list,
    max_tokens: int,
) -> list[SentenceGroup]:
    """Hard-split groups exceeding max_tokens at sentence boundaries."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    result: list[SentenceGroup] = []
    for group in groups:
        if group.tokens <= max_tokens:
            result.append(group)
            continue
        result.extend(_split_oversized(group, sentences, max_tokens))
    return result


def _split_oversized(
    group: SentenceGroup,
    sentences: list,
    max_tokens: int,
) -> list[SentenceGroup]:
    parts: list[SentenceGroup] = []
    start = group.start_sentence
    current_texts: list[str] = []
    current_tokens = 0

    for idx in range(group.start_sentence, group.end_sentence + 1):
        sent = sentences[idx]
        sent_tokens = token_count(sent.text)
        if sent_tokens > max_tokens:
            if current_texts:
                parts.append(_group(sentences, start, idx - 1))
                current_texts = []
                current_tokens = 0
            parts.append(
                SentenceGroup(
                    start_sentence=idx,
                    end_sentence=idx,
                    text=sent.text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                    oversized=True,
                )
            )
            start = idx + 1
            continue

        if current_texts and current_tokens + sent_tokens > max_tokens:
            parts.append(_group(sentences, start, idx - 1))
            start = idx
            current_texts = [sent.text]
            current_tokens = sent_tokens
        else:
            current_texts.append(sent.text)
            current_tokens += sent_tokens

    if current_texts:
        parts.append(_group(sentences, start, group.end_sentence))
    return parts


def merge_min_size(
    groups: list[SentenceGroup],
    sentences: list,
    min_tokens: int,
    max_tokens: int,
) -> list[SentenceGroup]:
    """Merge undersized adjacent groups when the merge fits under max_tokens."""
    if min_tokens < 0:
        raise ValueError("min_tokens must be >= 0")
    if not groups:
        return []

    merged: list[SentenceGroup] = [groups[0]]
    for group in groups[1:]:
        prev = merged[-1]
        if prev.tokens < min_tokens:
            combined_start = prev.start_sentence
            combined_end = group.end_sentence
            combined = _group(sentences, combined_start, combined_end)
            if combined.tokens <= max_tokens:
                merged[-1] = combined
                continue
        merged.append(group)

    # Trailing undersized: try merge backward.
    if len(merged) >= 2 and merged[-1].tokens < min_tokens:
        prev = merged[-2]
        combined = _group(
            sentences, prev.start_sentence, merged[-1].end_sentence
        )
        if combined.tokens <= max_tokens:
            merged[-2] = combined
            merged.pop()

    return merged

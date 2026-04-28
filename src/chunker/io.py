"""File I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from chunker.models import Chunk


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_jsonl(path: Path | None, chunks: Iterable[Chunk]) -> None:
    lines = [json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks]
    body = "\n".join(lines)
    if body:
        body += "\n"
    if path is None:
        print(body, end="")
    else:
        path.write_text(body, encoding="utf-8")

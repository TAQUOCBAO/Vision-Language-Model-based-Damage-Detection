"""Remove duplicate sentences from damage descriptions."""

from __future__ import annotations

import re


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def dedupe_sentences(text: str) -> str:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(part)
    return " ".join(kept)

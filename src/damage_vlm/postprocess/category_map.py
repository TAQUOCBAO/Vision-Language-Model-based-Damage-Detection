"""Map free-form category strings onto the closed label set."""

from __future__ import annotations

from typing import Any, Iterable

from damage_vlm.config import closed_labels, load_labels_config, synonyms_longest_first
from damage_vlm.data.silver_label import labels_from_text


def map_category_token(token: str, cfg: dict[str, Any] | None = None) -> str | None:
    cfg = cfg or load_labels_config()
    closed = set(closed_labels(cfg))
    raw = token.strip().lower().replace(" ", "_")
    if raw in closed:
        return raw
    # Also accept hyphen variants
    raw2 = raw.replace("-", "_")
    if raw2 in closed:
        return raw2
    hits = labels_from_text(token, cfg)
    return hits[0] if hits else None


def map_categories(categories: Iterable[str], cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_labels_config()
    closed = closed_labels(cfg)
    found: set[str] = set()
    for token in categories:
        mapped = map_category_token(str(token), cfg)
        if mapped:
            found.add(mapped)
    return [label for label in closed if label in found]

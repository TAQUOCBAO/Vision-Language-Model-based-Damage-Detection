"""Silver multi-label assignment from filename prefix and description keywords."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from damage_vlm.config import load_labels_config, prefix_seeds, synonyms_longest_first


_PREFIX_RE = re.compile(r"^([A-Za-z]+)_")


def extract_file_prefix(img: str) -> str | None:
    name = Path(img).name
    match = _PREFIX_RE.match(name)
    if not match:
        return None
    return match.group(1).lower()


def labels_from_prefix(img: str, cfg: dict[str, Any] | None = None) -> list[str]:
    seeds = prefix_seeds(cfg)
    prefix = extract_file_prefix(img)
    if not prefix:
        return []
    return list(seeds.get(prefix, []))


def labels_from_text(text: str, cfg: dict[str, Any] | None = None) -> list[str]:
    """Match synonyms longest-first; assign each closed label at most once."""
    cfg = cfg or load_labels_config()
    syn = synonyms_longest_first(cfg)
    lowered = text.lower()
    found: list[str] = []
    for label, phrases in syn.items():
        for phrase in phrases:
            if phrase.lower() in lowered:
                found.append(label)
                break
    return found


def silver_labels_for_image(
    img: str,
    description: str,
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Union of prefix seeds and text keyword hits, stable closed-set order."""
    cfg = cfg or load_labels_config()
    closed = list(cfg["closed_labels"])
    combined = set(labels_from_prefix(img, cfg)) | set(labels_from_text(description, cfg))
    ordered = [label for label in closed if label in combined]
    if not ordered:
        # Fall back to text-only scan again; if still empty, leave empty for caller to handle
        ordered = [label for label in closed if label in set(labels_from_text(description, cfg))]
    return ordered

"""Index external damage datasets mounted under data/external/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_external_index(path: Path | str) -> list[dict[str, Any]]:
    """Load a JSON list of {image_path, damage_categories} entries."""
    path = Path(path)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"External index must be a list: {path}")
    return data


def filter_rare_classes(
    rows: list[dict[str, Any]],
    rare: set[str] | None = None,
    limit: int = 800,
) -> list[dict[str, Any]]:
    rare = rare or {"potholes", "honeycomb", "efflorescence", "looseness", "peeling"}
    preferred = [r for r in rows if set(r.get("damage_categories", [])) & rare]
    others = [r for r in rows if r not in preferred]
    ordered = preferred + others
    return ordered[:limit]

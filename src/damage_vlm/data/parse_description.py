"""Parse competition description.json (line-delimited objects inside brackets)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_description_json(path: Path | str) -> list[dict[str, Any]]:
    """Load entries from a non-strict JSON array of one-object-per-line records."""
    path = Path(path)
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip().rstrip(",")
            if not s or s in ("[", "]"):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON object at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            for key in ("img", "prompt", "label"):
                if key not in obj:
                    raise ValueError(f"Missing '{key}' at {path}:{line_no}")
            entries.append(obj)
    return entries


def group_by_image(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entries:
        grouped[str(row["img"])].append(row)
    return dict(grouped)


def longest_label(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("No rows for image")
    return max((str(r["label"]) for r in rows), key=len)


def image_id_from_img_path(img: str) -> str:
    return Path(img).stem

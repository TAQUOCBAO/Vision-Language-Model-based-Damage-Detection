"""Folder inference helpers (schema-focused; model backend pluggable)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from damage_vlm.postprocess.pipeline import postprocess_prediction


GenerateFn = Callable[[Path], str]


def predict_folder(
    image_dir: Path | str,
    generate_fn: GenerateFn,
    *,
    extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
) -> list[dict[str, Any]]:
    image_dir = Path(image_dir)
    paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix in set(extensions)
    )
    outputs: list[dict[str, Any]] = []
    for path in paths:
        raw = generate_fn(path)
        outputs.append(postprocess_prediction(raw, default_image_id=path.stem))
    return outputs


def write_predictions(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

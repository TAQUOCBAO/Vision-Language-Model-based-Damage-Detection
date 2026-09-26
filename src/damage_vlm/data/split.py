"""Frozen multi-label stratified train/val split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from damage_vlm.config import closed_labels, load_eval_config


def multilabel_matrix(
    image_ids: Sequence[str],
    labels_by_id: dict[str, list[str]],
) -> np.ndarray:
    classes = closed_labels()
    index = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(image_ids), len(classes)), dtype=np.int8)
    for row, image_id in enumerate(image_ids):
        for label in labels_by_id.get(image_id, []):
            if label in index:
                y[row, index[label]] = 1
    # Ensure every row has at least one positive for stratification stability
    empty = y.sum(axis=1) == 0
    if empty.any():
        y[empty, 0] = 1
    return y


def make_split(
    image_ids: Sequence[str],
    labels_by_id: dict[str, list[str]],
    *,
    seed: int | None = None,
    test_size: float | None = None,
) -> dict[str, Any]:
    eval_cfg = load_eval_config()
    seed = int(eval_cfg["split"]["seed"] if seed is None else seed)
    test_size = float(eval_cfg["split"]["test_size"] if test_size is None else test_size)

    ids = list(image_ids)
    y = multilabel_matrix(ids, labels_by_id)
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(np.zeros(len(ids)), y))
    train_ids = [ids[i] for i in train_idx]
    val_ids = [ids[i] for i in val_idx]
    return {
        "version": "v1",
        "seed": seed,
        "test_size": test_size,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
    }


def save_split(split: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_split(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

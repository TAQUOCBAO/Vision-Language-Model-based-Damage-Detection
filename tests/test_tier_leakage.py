"""Tier train manifest must not include frozen val IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from damage_vlm.config import REPO_ROOT


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "processed" / "splits" / "v1.json").is_file(),
    reason="processed split not built yet",
)
def test_tier_a_train_no_val_leak() -> None:
    split = json.loads((REPO_ROOT / "data" / "processed" / "splits" / "v1.json").read_text())
    val = set(split["val_ids"])
    path = REPO_ROOT / "data" / "processed" / "tiers" / "a" / "train.jsonl"
    ids = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["image_id"])
    assert set(ids).isdisjoint(val)

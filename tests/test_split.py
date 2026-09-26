"""Frozen multilabel split tests."""

from __future__ import annotations

from damage_vlm.data.split import make_split


def test_split_sizes_and_disjoint() -> None:
    # Synthetic 20 samples with two labels
    labels_by_id = {}
    for i in range(20):
        labels_by_id[f"id_{i:02d}"] = ["cracks"] if i % 2 == 0 else ["voids"]
    split = make_split(sorted(labels_by_id), labels_by_id, seed=42, test_size=0.2)
    assert len(split["train_ids"]) == 16
    assert len(split["val_ids"]) == 4
    assert set(split["train_ids"]).isdisjoint(split["val_ids"])


def test_split_deterministic() -> None:
    labels_by_id = {f"id_{i:02d}": ["cracks", "spalling"][i % 2 : i % 2 + 1] for i in range(30)}
    # ensure non-empty labels
    labels_by_id = {
        f"id_{i:02d}": (["cracks"] if i % 2 == 0 else ["spalling"]) for i in range(30)
    }
    a = make_split(sorted(labels_by_id), labels_by_id, seed=7, test_size=0.2)
    b = make_split(sorted(labels_by_id), labels_by_id, seed=7, test_size=0.2)
    assert a["train_ids"] == b["train_ids"]
    assert a["val_ids"] == b["val_ids"]

"""Official test-set inventory, prompt compliance, and scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from damage_vlm.config import load_prompts_config
from damage_vlm.data.build_sft import build_official_user_prompt
from damage_vlm.data.official_test import (
    DEFAULT_OFFICIAL_IMAGE_DIR,
    OFFICIAL_TEST_COUNT,
    OfficialTestInventoryError,
    list_official_test_images,
    load_official_gold,
)
from damage_vlm.metrics.official import official_scores, subset_accuracy


def test_official_prompt_contains_exact_q1_q2() -> None:
    prompts = load_prompts_config()
    q1 = "Q1: Determine whether there is structural damage in the image?"
    q2 = "Q2: Describe the damage characteristics based on the image?"
    assert prompts["official_q1"].strip() == q1
    assert prompts["official_q2"].strip() == q2
    text = build_official_user_prompt("001")
    assert q1 in text
    assert q2 in text
    assert text.count("<image>") == 1
    assert "001" in text


def test_official_inventory_rejects_incomplete_folder(tmp_path: Path) -> None:
    (tmp_path / "001.jpg").write_bytes(b"x")
    with pytest.raises(OfficialTestInventoryError):
        list_official_test_images(tmp_path)


def test_official_gold_loader_and_scores(tmp_path: Path) -> None:
    gold = {
        "001": {
            "damage_categories": ["cracks"],
            "description": "A vertical crack is visible on the wall.",
        },
        "002": {
            "damage_categories": [],
            "description": "No apparent structural damage.",
            "has_structural_damage": False,
        },
    }
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(gold), encoding="utf-8")
    loaded = load_official_gold(path)
    assert loaded["001"]["has_structural_damage"] is True
    assert loaded["002"]["has_structural_damage"] is False

    preds_cats = [["cracks"], []]
    hyps = [
        "A vertical crack is visible on the wall.",
        "No apparent structural damage.",
    ]
    scores = official_scores(
        [loaded["001"]["damage_categories"], loaded["002"]["damage_categories"]],
        preds_cats,
        [loaded["001"]["description"], loaded["002"]["description"]],
        hyps,
        true_has_damage=[loaded["001"]["has_structural_damage"], loaded["002"]["has_structural_damage"]],
        pred_has_damage=[True, False],
    )
    assert scores["n"] == 2
    assert scores["q1_binary_accuracy"] == 1.0
    assert scores["category_subset_accuracy"] == 1.0
    assert scores["category_label_accuracy"] == 1.0
    assert scores["meteor"] > 0.9


def test_subset_accuracy_partial_mismatch() -> None:
    assert subset_accuracy([["cracks", "spalling"]], [["cracks"]]) == 0.0


@pytest.mark.skipif(not DEFAULT_OFFICIAL_IMAGE_DIR.is_dir(), reason="official test images not present")
def test_live_official_folder_has_110_images() -> None:
    paths = list_official_test_images()
    assert len(paths) == OFFICIAL_TEST_COUNT
    assert paths[0].stem == "001"
    assert paths[-1].stem == "110"

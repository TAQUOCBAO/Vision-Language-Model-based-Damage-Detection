"""Tests for multimodal SFT row construction."""

from __future__ import annotations

from damage_vlm.data.build_sft import build_sft_row, build_user_prompt


def test_user_prompt_includes_image_placeholder() -> None:
    text = build_user_prompt("00002")
    assert text.count("<image>") == 1
    assert "00002" in text


def test_sft_row_image_token_matches_images() -> None:
    row = build_sft_row(
        image_id="00002",
        image_path="data/dataset/image/00002.jpg",
        damage_categories=["cracks"],
        description="A crack is visible.",
    )
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    assert user.count("<image>") == len(row["images"]) == 1

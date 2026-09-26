"""Tests for description parsing and image path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from damage_vlm.config import REPO_ROOT
from damage_vlm.data.parse_description import (
    group_by_image,
    longest_label,
    parse_description_json,
)
from damage_vlm.data.resolve_image import resolve_image_path

DATASET = REPO_ROOT / "data" / "dataset"


def test_parse_description_counts() -> None:
    entries = parse_description_json(DATASET / "description.json")
    assert len(entries) == 2400
    grouped = group_by_image(entries)
    assert len(grouped) == 1200


def test_longest_label_prefers_description() -> None:
    rows = [
        {"img": "image/x.jpg", "prompt": "p", "label": "Yes, there is a crack."},
        {
            "img": "image/x.jpg",
            "prompt": "p2",
            "label": "There is a crack on the concrete surface extending from left to right.",
        },
    ]
    assert "extending" in longest_label(rows)


def test_resolve_existing_crack_image() -> None:
    path = resolve_image_path(DATASET, "image/crack_0001.jpg")
    assert path.is_file()


def test_resolve_extension_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "image"
    image_dir.mkdir()
    real = image_dir / "19.png"
    real.write_bytes(b"fake")
    resolved = resolve_image_path(tmp_path, "image/19.jpg")
    assert resolved == real.resolve()

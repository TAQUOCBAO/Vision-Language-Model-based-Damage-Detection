"""Official hidden-test inventory and gold-label loaders.

Complies with ``Test Requirements.docx``:
110 images named ``001.jpg`` … ``110.jpg``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from damage_vlm.config import REPO_ROOT, closed_labels

OFFICIAL_TEST_COUNT = 110
OFFICIAL_IMAGE_STEMS = [f"{i:03d}" for i in range(1, OFFICIAL_TEST_COUNT + 1)]
DEFAULT_OFFICIAL_IMAGE_DIR = (
    REPO_ROOT / "data_v2" / "Dataset-Project 3-Test" / "dataset" / "image"
)
DEFAULT_OFFICIAL_REQUIREMENTS = (
    REPO_ROOT / "data_v2" / "Dataset-Project 3-Test" / "dataset" / "Test Requirements.docx"
)


class OfficialTestInventoryError(ValueError):
    """Raised when the official test folder does not match the spec."""


def official_image_path(image_dir: Path | str, stem: str) -> Path:
    image_dir = Path(image_dir)
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise OfficialTestInventoryError(f"Missing official test image: {stem}.jpg")


def list_official_test_images(image_dir: Path | str | None = None) -> list[Path]:
    """Return the 110 official test images in spec order (001 … 110)."""
    image_dir = Path(image_dir) if image_dir else DEFAULT_OFFICIAL_IMAGE_DIR
    if not image_dir.is_dir():
        raise OfficialTestInventoryError(f"Official test image directory not found: {image_dir}")

    missing: list[str] = []
    paths: list[Path] = []
    for stem in OFFICIAL_IMAGE_STEMS:
        try:
            paths.append(official_image_path(image_dir, stem))
        except OfficialTestInventoryError:
            missing.append(f"{stem}.jpg")
    extras = sorted(
        p.name
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stem not in OFFICIAL_IMAGE_STEMS
    )
    if missing:
        raise OfficialTestInventoryError(
            f"Official test set must contain 001.jpg–110.jpg. Missing {len(missing)} file(s), "
            f"e.g. {missing[:5]}"
        )
    if extras:
        raise OfficialTestInventoryError(
            f"Official test set has unexpected extra images: {extras[:5]}"
        )
    return paths


def load_official_gold(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load organizer (or local) gold labels.

    Accepted shapes:
    - JSON object keyed by ``image_id``
    - JSON list of objects with ``image_id``
    Each record needs ``damage_categories`` (list) and ``description`` (string).
    Optional ``has_structural_damage`` (bool) overrides Q1 gold.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]]
    if isinstance(raw, dict):
        if "image_id" in raw and "damage_categories" in raw:
            rows = [raw]
        else:
            rows = []
            for key, value in raw.items():
                if not isinstance(value, dict):
                    raise ValueError(f"Gold entry {key!r} must be an object")
                row = dict(value)
                row.setdefault("image_id", str(key))
                rows.append(row)
    elif isinstance(raw, list):
        rows = [r for r in raw if isinstance(r, dict)]
    else:
        raise ValueError(f"Gold labels must be a JSON object or list: {path}")

    allowed = set(closed_labels())
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = str(row.get("image_id") or "").strip()
        if not image_id:
            raise ValueError("Gold record missing image_id")
        cats = row.get("damage_categories") or []
        if isinstance(cats, str):
            cats = [cats]
        cats = [str(c) for c in cats if str(c) in allowed]
        description = str(row.get("description") or "")
        if "has_structural_damage" in row:
            has_damage = bool(row["has_structural_damage"])
        else:
            has_damage = bool(cats)
        out[image_id] = {
            "image_id": image_id,
            "damage_categories": cats,
            "description": description,
            "has_structural_damage": has_damage,
        }
    return out

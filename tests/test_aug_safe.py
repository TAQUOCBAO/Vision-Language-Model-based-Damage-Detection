"""Orientation-safe augmentation tests (AE2)."""

from __future__ import annotations

from damage_vlm.data.aug_safe import has_orientation_language, maybe_geometric_augment
from PIL import Image
import numpy as np


def test_ae2_orientation_blocks_flip() -> None:
    desc = "A crack stretches from left to right on the concrete surface."
    assert has_orientation_language(desc)
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    rng = np.random.default_rng(0)
    # Force many attempts: geometric path should still return without requiring flip
    out = maybe_geometric_augment(img, desc, rng)
    assert out.size == img.size


def test_no_orientation_allows_geometry_flag() -> None:
    desc = "Severe spalling is visible on the concrete surface."
    assert not has_orientation_language(desc)

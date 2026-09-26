"""Orientation-safe image augmentation for Tier B."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

_ORIENTATION_RE = re.compile(
    r"left\s*to\s*right|right\s*to\s*left|top\s*to\s*bottom|bottom\s*to\s*top|"
    r"vertical|horizontal|diagonal|top-left|top-right|bottom-left|bottom-right|"
    r"from the left|from the right|upward|downward",
    re.IGNORECASE,
)


def has_orientation_language(description: str) -> bool:
    return bool(_ORIENTATION_RE.search(description))


def photometric_augment(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    img = image.convert("RGB")
    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.75, 1.25)))
    if rng.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.75, 1.25)))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.2))))
    if rng.random() < 0.3:
        arr = np.asarray(img).astype(np.float32)
        noise = rng.normal(0, rng.uniform(3.0, 12.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def maybe_geometric_augment(
    image: Image.Image,
    description: str,
    rng: np.random.Generator,
) -> Image.Image:
    """Apply mild geometric aug only when description has no orientation language."""
    if has_orientation_language(description):
        return image
    img = image
    if rng.random() < 0.3:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.15:
        img = img.rotate(float(rng.uniform(-8, 8)), expand=False, fillcolor=(0, 0, 0))
    return img


def augment_image(
    image: Image.Image,
    description: str,
    *,
    seed: int,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = photometric_augment(image, rng)
    img = maybe_geometric_augment(img, description, rng)
    return img


def assert_no_flip_when_oriented(description: str, applied_ops: list[str]) -> None:
    if has_orientation_language(description) and "flip_h" in applied_ops:
        raise AssertionError("Horizontal flip forbidden when orientation language is present")

"""Resolve image paths when extensions or case differ from JSON references."""

from __future__ import annotations

from pathlib import Path

_ALT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp", ".WEBP")


def resolve_image_path(dataset_root: Path | str, img_rel: str) -> Path:
    """Resolve ``img`` relative to dataset root, trying alternate extensions/case.

    Parameters
    ----------
    dataset_root:
        Directory that contains ``image/`` (typically ``data/dataset``).
    img_rel:
        Relative path from JSON, e.g. ``image/crack_0001.jpg``.
    """
    root = Path(dataset_root)
    candidate = root / img_rel
    if candidate.is_file():
        return candidate.resolve()

    stem_path = candidate.with_suffix("")
    parent = candidate.parent
    stem = candidate.stem

    # Exact stem with alternate extensions
    for ext in _ALT_EXTENSIONS:
        alt = parent / f"{stem}{ext}"
        if alt.is_file():
            return alt.resolve()

    # Case-insensitive match within the parent directory
    if parent.is_dir():
        lower_stem = stem.lower()
        for p in parent.iterdir():
            if not p.is_file():
                continue
            if p.stem.lower() == lower_stem:
                return p.resolve()

    raise FileNotFoundError(f"Could not resolve image for '{img_rel}' under {root}")

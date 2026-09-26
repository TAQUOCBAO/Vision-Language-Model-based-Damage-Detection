"""Normalize descriptions toward a stable technical pattern without inventing labels."""

from __future__ import annotations

import re

from damage_vlm.data.silver_label import labels_from_text


def normalize_description(description: str, categories: list[str]) -> str:
    text = " ".join(description.strip().split())
    # Ensure known categories remain mentioned; do not invent new ones.
    mentioned = set(labels_from_text(text))
    for label in categories:
        if label not in mentioned:
            # Light touch: append canonical label token once for METEOR stability
            pretty = label.replace("_", " ")
            text = f"{text} Additional {pretty} is present."
            mentioned.add(label)
    # Drop trailing duplicate spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

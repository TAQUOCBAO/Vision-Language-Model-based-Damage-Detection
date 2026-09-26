"""Extract and repair JSON objects from model text."""

from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model output, tolerating fences and chatter."""
    text = text.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not extract a JSON object from model output")


def normalize_prediction(obj: dict[str, Any], *, default_image_id: str = "") -> dict[str, Any]:
    image_id = str(obj.get("image_id") or default_image_id)
    cats = obj.get("damage_categories") or []
    if isinstance(cats, str):
        cats = [cats]
    description = str(obj.get("description") or "")
    return {
        "image_id": image_id,
        "damage_categories": list(cats),
        "description": description,
    }

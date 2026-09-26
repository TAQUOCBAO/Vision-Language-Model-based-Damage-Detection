"""Full post-process pipeline applied before metrics and submission."""

from __future__ import annotations

from typing import Any

from damage_vlm.postprocess.align import align_description
from damage_vlm.postprocess.category_map import map_categories
from damage_vlm.postprocess.dedupe import dedupe_sentences
from damage_vlm.postprocess.json_repair import extract_json_object, normalize_prediction


def postprocess_prediction(
    raw_text: str,
    *,
    default_image_id: str = "",
) -> dict[str, Any]:
    obj = normalize_prediction(extract_json_object(raw_text), default_image_id=default_image_id)
    categories = map_categories(obj["damage_categories"])
    # Also mine description for extra closed labels if model omitted them from the list
    from damage_vlm.data.silver_label import labels_from_text

    categories = map_categories(list(categories) + labels_from_text(obj["description"]))
    description = align_description(obj["description"], categories)
    description = dedupe_sentences(description)
    image_id = obj["image_id"] or default_image_id
    return {
        "image_id": image_id,
        "damage_categories": categories,
        "description": description,
    }

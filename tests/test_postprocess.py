"""Post-process tests covering AE4 and synonym mapping."""

from __future__ import annotations

from damage_vlm.postprocess.align import align_description
from damage_vlm.postprocess.category_map import map_categories
from damage_vlm.postprocess.dedupe import dedupe_sentences
from damage_vlm.postprocess.pipeline import postprocess_prediction


def test_ae4_alignment_appends_corrosion_template() -> None:
    text = align_description(
        "Multiple vertical concrete cracks observed on the column surface.",
        ["cracks", "corrosion"],
    )
    assert "crack" in text.lower()
    assert "corrosion" in text.lower()


def test_dedupe_removes_duplicate_sentence() -> None:
    text = "Cracks are visible. Cracks are visible. Spalling is minor."
    out = dedupe_sentences(text)
    assert out.count("Cracks are visible.") == 1
    assert "Spalling" in out


def test_fissure_category_maps_to_cracks() -> None:
    assert map_categories(["fissure", "rust"]) == ["cracks", "corrosion"]


def test_pipeline_repairs_fenced_json() -> None:
    raw = """Sure, here is the result:
```json
{"image_id": "demo", "damage_categories": ["cracks", "corrosion"], "description": "Vertical cracks on the column."}
```
"""
    out = postprocess_prediction(raw)
    assert out["image_id"] == "demo"
    assert "cracks" in out["damage_categories"]
    assert "corrosion" in out["damage_categories"]
    assert "corrosion" in out["description"].lower()

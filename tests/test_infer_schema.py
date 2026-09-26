"""Inference / submission schema tests."""

from __future__ import annotations

import json
from pathlib import Path

from damage_vlm.config import closed_labels
from damage_vlm.infer.predict_folder import predict_folder
from damage_vlm.postprocess.pipeline import postprocess_prediction


def test_fenced_output_becomes_schema_valid() -> None:
    raw = """```json
{"image_id": "x", "damage_categories": ["fissure"], "description": "A fissure is visible."}
```"""
    out = postprocess_prediction(raw, default_image_id="x")
    assert set(out.keys()) == {"image_id", "damage_categories", "description"}
    assert out["damage_categories"] == ["cracks"]
    assert all(c in closed_labels() for c in out["damage_categories"])


def test_predict_folder_fixture(tmp_path: Path) -> None:
    img = tmp_path / "sample.jpg"
    img.write_bytes(b"not-a-real-image")

    def gen(path: Path) -> str:
        return json.dumps(
            {
                "image_id": path.stem,
                "damage_categories": ["cracks"],
                "description": "Cracks observed.",
            }
        )

    rows = predict_folder(tmp_path, gen)
    assert len(rows) == 1
    assert rows[0]["image_id"] == "sample"

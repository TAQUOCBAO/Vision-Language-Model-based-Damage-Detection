"""Submission schema tests."""

from __future__ import annotations

import json
from pathlib import Path

from damage_vlm.config import closed_labels
from damage_vlm.infer.predict_folder import predict_folder, write_predictions


def test_submission_schema(tmp_path: Path) -> None:
    img = tmp_path / "00002.jpg"
    img.write_bytes(b"x")

    def gen(path: Path) -> str:
        return json.dumps(
            {
                "image_id": path.stem,
                "damage_categories": ["cracks", "spalling"],
                "description": "Cracks and spalling are visible.",
            }
        )

    rows = predict_folder(tmp_path, gen)
    out = tmp_path / "submission.json"
    write_predictions(out, rows)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    for row in loaded:
        assert set(row.keys()) >= {"image_id", "damage_categories", "description"}
        assert row["image_id"] == "00002"
        assert set(row["damage_categories"]) <= set(closed_labels())

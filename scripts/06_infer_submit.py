#!/usr/bin/env python3
"""Produce competition submission JSON for an image folder.

Uses a pluggable generator. For dry schema checks, --fixture-mode invents
JSON from filenames without loading a VLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from damage_vlm.config import REPO_ROOT, closed_labels
from damage_vlm.infer.predict_folder import predict_folder, write_predictions
from damage_vlm.postprocess.pipeline import postprocess_prediction


def fixture_generate(path: Path) -> str:
    payload = {
        "image_id": path.stem,
        "damage_categories": ["cracks"],
        "description": f"Apparent cracks observed in image {path.stem}.",
    }
    return json.dumps(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "submissions" / "submission.json",
    )
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="Generate placeholder predictions without a VLM (schema smoke only).",
    )
    args = parser.parse_args()

    if not args.fixture_mode:
        raise SystemExit(
            "Real VLM inference is not wired in this script yet. "
            "Use --fixture-mode for schema smoke, or plug generate_fn in predict_folder."
        )

    rows = predict_folder(args.image_dir, fixture_generate)
    # Ensure categories ⊆ closed set
    allowed = set(closed_labels())
    for row in rows:
        row["damage_categories"] = [c for c in row["damage_categories"] if c in allowed]
        fixed = postprocess_prediction(json.dumps(row), default_image_id=row["image_id"])
        row.update(fixed)

    write_predictions(args.out, rows)
    print(f"Wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()

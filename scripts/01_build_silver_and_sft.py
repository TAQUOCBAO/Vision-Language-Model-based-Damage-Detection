#!/usr/bin/env python3
"""Build silver labels, frozen split, and Tier A SFT datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from damage_vlm.config import REPO_ROOT
from damage_vlm.data.build_sft import build_sft_row
from damage_vlm.data.parse_description import (
    group_by_image,
    image_id_from_img_path,
    longest_label,
    parse_description_json,
)
from damage_vlm.data.resolve_image import resolve_image_path
from damage_vlm.data.silver_label import silver_labels_for_image
from damage_vlm.data.split import make_split, save_split


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "dataset",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
    )
    args = parser.parse_args()

    desc_path = args.dataset_root / "description.json"
    entries = parse_description_json(desc_path)
    grouped = group_by_image(entries)

    records: list[dict] = []
    silver: dict[str, dict] = {}
    labels_by_id: dict[str, list[str]] = {}

    for img_rel, rows in grouped.items():
        image_id = image_id_from_img_path(img_rel)
        description = longest_label(rows)
        categories = silver_labels_for_image(img_rel, description)
        abs_path = resolve_image_path(args.dataset_root, img_rel)
        # Store path relative to repo for portability in SFT jsonl
        try:
            rel_for_sft = str(abs_path.relative_to(REPO_ROOT))
        except ValueError:
            rel_for_sft = str(abs_path)
        row = build_sft_row(
            image_id=image_id,
            image_path=rel_for_sft,
            damage_categories=categories,
            description=description,
        )
        records.append(row)
        labels_by_id[image_id] = categories
        silver[image_id] = {
            "image_id": image_id,
            "img": img_rel,
            "image_path": rel_for_sft,
            "damage_categories": categories,
            "description": description,
        }

    processed = args.processed_root
    processed.mkdir(parents=True, exist_ok=True)
    silver_path = processed / "silver_labels.json"
    silver_path.write_text(json.dumps(silver, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    image_ids = sorted(labels_by_id.keys())
    split = make_split(image_ids, labels_by_id)
    split_path = processed / "splits" / "v1.json"
    save_split(split, split_path)

    by_id = {r["image_id"]: r for r in records}
    train_rows = [by_id[i] for i in split["train_ids"]]
    val_rows = [by_id[i] for i in split["val_ids"]]

    tier_a = processed / "tiers" / "a"
    write_jsonl(tier_a / "train.jsonl", train_rows)
    write_jsonl(tier_a / "val.jsonl", val_rows)

    print(
        f"Built silver={len(silver)} train={len(train_rows)} val={len(val_rows)} "
        f"split={split_path}"
    )


if __name__ == "__main__":
    main()

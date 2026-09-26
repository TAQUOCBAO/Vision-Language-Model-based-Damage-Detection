#!/usr/bin/env python3
"""Build full-corpus SFT for the winning tier on an updated dataset (e.g. data_v2).

Uses ``outputs/evals/winner.json`` (default) to pick the recipe:
  - tier a: base silver SFT on all images
  - tier b: base + aug/normalize expansion on all images
  - tier c: base + paraphrase expansion on all images

Writes ``data/processed/full_winner/train.jsonl`` for LLaMA-Factory
``damage_full_winner`` (see ``configs/llamafactory/full_winner_lora_sft.yaml``).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from damage_vlm.config import REPO_ROOT
from damage_vlm.data.aug_safe import augment_image
from damage_vlm.data.build_sft import build_sft_row
from damage_vlm.data.normalize_desc import normalize_description
from damage_vlm.data.paraphrase import identity_paraphrase_fn, paraphrase_n
from damage_vlm.data.parse_description import (
    group_by_image,
    image_id_from_img_path,
    longest_label,
    parse_description_json,
)
from damage_vlm.data.resolve_image import resolve_image_path
from damage_vlm.data.silver_label import silver_labels_for_image


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_base_rows(dataset_root: Path) -> tuple[list[dict], dict[str, dict]]:
    entries = parse_description_json(dataset_root / "description.json")
    grouped = group_by_image(entries)
    records: list[dict] = []
    silver: dict[str, dict] = {}
    for img_rel, rows in grouped.items():
        image_id = image_id_from_img_path(img_rel)
        description = longest_label(rows)
        categories = silver_labels_for_image(img_rel, description)
        abs_path = resolve_image_path(dataset_root, img_rel)
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
        silver[image_id] = {
            "image_id": image_id,
            "img": img_rel,
            "image_path": rel_for_sft,
            "damage_categories": categories,
            "description": description,
        }
    return records, silver


def expand_tier_b(base_rows: list[dict], out_img: Path, seed: int = 42) -> list[dict]:
    out_img.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict] = []
    for i, row in enumerate(base_rows):
        src = REPO_ROOT / row["images"][0]
        description = normalize_description(row["description"], row["damage_categories"])
        img = Image.open(src)
        aug = augment_image(img, description, seed=seed + i)
        dest = out_img / f"{row['image_id']}_aug.jpg"
        aug.save(dest, quality=95)
        rel = str(dest.relative_to(REPO_ROOT))
        new_rows.append(
            build_sft_row(
                image_id=row["image_id"],
                image_path=rel,
                damage_categories=row["damage_categories"],
                description=description,
            )
        )
        new_rows.append(row)
    return new_rows


def expand_tier_c(base_rows: list[dict], paraphrase_times: int = 3) -> list[dict]:
    expanded: list[dict] = list(base_rows)
    for row in base_rows:
        for paraphrased in paraphrase_n(
            row["description"], paraphrase_times, identity_paraphrase_fn
        ):
            expanded.append(
                build_sft_row(
                    image_id=row["image_id"],
                    image_path=row["images"][0],
                    damage_categories=row["damage_categories"],
                    description=paraphrased,
                )
            )
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data_v2" / "dataset",
        help="Updated corpus root (description.json + image/)",
    )
    parser.add_argument(
        "--winner-json",
        type=Path,
        default=REPO_ROOT / "outputs" / "evals" / "winner.json",
    )
    parser.add_argument(
        "--tier",
        choices=["a", "b", "c"],
        default=None,
        help="Override winner tier (default: read from --winner-json)",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output train.jsonl (default: <processed-root>/full_winner/train.jsonl)",
    )
    args = parser.parse_args()

    if args.tier is None:
        if not args.winner_json.is_file():
            raise SystemExit(
                f"Missing {args.winner_json}; pass --tier explicitly or run 05_select_winner.py"
            )
        winner = json.loads(args.winner_json.read_text(encoding="utf-8"))
        tier = str(winner.get("winner_tier") or winner.get("winner", {}).get("tier"))
        if tier not in {"a", "b", "c"}:
            raise SystemExit(f"Invalid winner tier in {args.winner_json}: {tier!r}")
    else:
        tier = args.tier

    out = args.out or (args.processed_root / "full_winner" / "train.jsonl")
    silver_path = args.processed_root / "full_winner" / "silver_labels.json"

    print(f"Building full-winner SFT: tier={tier} dataset={args.dataset_root}")
    base_rows, silver = build_base_rows(args.dataset_root)
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    silver_path.write_text(json.dumps(silver, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if tier == "a":
        train_rows = base_rows
    elif tier == "b":
        train_rows = expand_tier_b(
            base_rows, args.processed_root / "full_winner" / "images"
        )
    else:
        train_rows = expand_tier_c(base_rows)

    # Stable order for reproducibility
    train_rows = sorted(train_rows, key=lambda r: (r["image_id"], r["description"]))
    if out.exists():
        # Drop stale aug dir only when rebuilding B
        pass
    write_jsonl(out, train_rows)

    meta = {
        "winner_tier": tier,
        "dataset_root": str(args.dataset_root),
        "n_images": len(silver),
        "n_train_rows": len(train_rows),
        "train_jsonl": str(out),
        "silver_json": str(silver_path),
    }
    meta_path = args.processed_root / "full_winner" / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("Next: python scripts/03_train_tier.py --tier full")


if __name__ == "__main__":
    main()

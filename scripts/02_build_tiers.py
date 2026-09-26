#!/usr/bin/env python3
"""Build Tier B (aug+normalize) and Tier C (paraphrase / external) datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from damage_vlm.config import REPO_ROOT
from damage_vlm.data.aug_safe import augment_image
from damage_vlm.data.build_sft import build_sft_row
from damage_vlm.data.external_index import filter_rare_classes, load_external_index
from damage_vlm.data.normalize_desc import normalize_description
from damage_vlm.data.paraphrase import identity_paraphrase_fn, paraphrase_n
from damage_vlm.data.pseudo_caption import build_pseudo_caption_prompt
from damage_vlm.data.split import load_split


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def assert_no_val_leak(rows: list[dict], val_ids: set[str]) -> None:
    leaked = [r["image_id"] for r in rows if r["image_id"] in val_ids]
    if leaked:
        raise RuntimeError(f"Val leakage detected in tier train set: {leaked[:5]}")


def build_tier_b(train_rows: list[dict], out_dir: Path, seed: int = 42) -> None:
    out_img = out_dir / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict] = []
    for i, row in enumerate(train_rows):
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
        # Keep original as well
        new_rows.append(row)
    write_jsonl(out_dir / "train.jsonl", new_rows)


def build_tier_c(
    train_rows: list[dict],
    out_dir: Path,
    *,
    paraphrase_times: int = 3,
    external_index: Path | None = None,
    use_identity_paraphrase: bool = True,
) -> None:
    generate_fn = identity_paraphrase_fn if use_identity_paraphrase else identity_paraphrase_fn
    # NOTE: replace generate_fn with a vLLM/local LLM callable for real Tier C runs.
    expanded: list[dict] = list(train_rows)
    for row in train_rows:
        for paraphrased in paraphrase_n(row["description"], paraphrase_times, generate_fn):
            expanded.append(
                build_sft_row(
                    image_id=row["image_id"],
                    image_path=row["images"][0],
                    damage_categories=row["damage_categories"],
                    description=paraphrased,
                )
            )

    if external_index and external_index.is_file():
        ext_rows = filter_rare_classes(load_external_index(external_index))
        for item in ext_rows:
            # Pseudo-caption text placeholder until local VLM is wired
            prompt = build_pseudo_caption_prompt(item.get("damage_categories", []))
            description = item.get("description") or (
                f"Observed damage types: {', '.join(item.get('damage_categories', []))}. {prompt[:80]}"
            )
            image_id = Path(item["image_path"]).stem
            expanded.append(
                build_sft_row(
                    image_id=f"ext_{image_id}",
                    image_path=item["image_path"],
                    damage_categories=item.get("damage_categories", []),
                    description=description,
                )
            )
    else:
        print("WARNING: no external index — Tier C uses paraphrase-only expansion")

    write_jsonl(out_dir / "train.jsonl", expanded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument(
        "--external-index",
        type=Path,
        default=REPO_ROOT / "data" / "external" / "index.json",
    )
    args = parser.parse_args()

    split = load_split(args.processed_root / "splits" / "v1.json")
    val_ids = set(split["val_ids"])
    train_rows = read_jsonl(args.processed_root / "tiers" / "a" / "train.jsonl")
    val_rows = read_jsonl(args.processed_root / "tiers" / "a" / "val.jsonl")

    tier_b = args.processed_root / "tiers" / "b"
    tier_c = args.processed_root / "tiers" / "c"
    if tier_b.exists():
        shutil.rmtree(tier_b)
    if tier_c.exists():
        shutil.rmtree(tier_c)

    build_tier_b(train_rows, tier_b)
    # Copy val unchanged for all tiers
    write_jsonl(tier_b / "val.jsonl", val_rows)
    build_tier_c(train_rows, tier_c, external_index=args.external_index)
    write_jsonl(tier_c / "val.jsonl", val_rows)

    for name, path in ("b", tier_b / "train.jsonl"), ("c", tier_c / "train.jsonl"):
        rows = read_jsonl(path)
        assert_no_val_leak(rows, val_ids)
        print(f"Tier {name}: train_rows={len(rows)}")


if __name__ == "__main__":
    main()

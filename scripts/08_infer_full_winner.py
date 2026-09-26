#!/usr/bin/env python3
"""Run folder or single-image inference with the full-winner LoRA adapter.

Default adapter: ``outputs/runs/full_winner_v2``
Default images:  ``data_v2/dataset/image`` (unless ``--image`` is set)
Default output:  ``outputs/submissions/submission.json``

Examples:
  # One image
  python scripts/08_infer_full_winner.py --image data_v2/dataset/image/00002.jpg

  # Several explicit paths
  python scripts/08_infer_full_winner.py --image path/a.jpg --image path/b.jpg

  # Full folder
  python scripts/08_infer_full_winner.py --image-dir data_v2/dataset/image

  # Low VRAM (bitsandbytes NF4; needs: pip install bitsandbytes)
  python scripts/08_infer_full_winner.py --load-in-4bit --image path/to.jpg

  # After scripts/09_export_low_vram.py
  python scripts/08_infer_full_winner.py --load-in-4bit \\
    --adapter outputs/exports/full_winner_v2_lora --image path/to.jpg

  # Smoke / resume
  python scripts/08_infer_full_winner.py --limit 2
  python scripts/08_infer_full_winner.py --resume --max-images 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from damage_vlm.config import REPO_ROOT, closed_labels
from damage_vlm.infer.predict_folder import write_predictions
from damage_vlm.infer.qwen_vl import Qwen3VLLoRAGenerator
from damage_vlm.postprocess.pipeline import postprocess_prediction

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(rows, list):
        return {}
    return {r["image_id"]: r for r in rows if isinstance(r, dict) and "image_id" in r}


def _list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS
    )


def _resolve_image_paths(raw_paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for raw in raw_paths:
        path = raw if raw.is_absolute() else (REPO_ROOT / raw)
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"Image file not found: {raw}")
        if path.suffix not in IMAGE_EXTS:
            raise SystemExit(f"Unsupported image extension: {path}")
        resolved.append(path)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPO_ROOT / "outputs" / "runs" / "full_winner_v2",
        help="LoRA adapter directory from full-winner train (or 09 export package)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Merged full model dir (from 09 --merge). Skips LoRA adapter load.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load backbone in bitsandbytes NF4 (much lower VRAM)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load backbone in bitsandbytes 8-bit",
    )
    parser.add_argument(
        "--image",
        "--image-path",
        dest="images",
        type=Path,
        action="append",
        default=None,
        help="Path to one image file (repeatable). Overrides --image-dir when set.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=REPO_ROOT / "data_v2" / "dataset" / "image",
        help="Directory of images (used when --image is not provided)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "submissions" / "submission.json",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap total images (smoke)")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process at most this many pending images then exit (chunked runs)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip ids already in --out")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Choose only one of --load-in-4bit / --load-in-8bit")
    quant = "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else "none"

    if args.model_dir is not None:
        if not args.model_dir.is_dir():
            raise SystemExit(f"Merged model dir not found: {args.model_dir}")
        adapter_dir = None
    else:
        if not (args.adapter / "adapter_config.json").is_file():
            raise SystemExit(
                f"Adapter not found under {args.adapter}\n"
                "Train first: python scripts/03_train_tier.py --tier full\n"
                "Or export: python scripts/09_export_low_vram.py"
            )
        adapter_dir = args.adapter

    if args.images:
        paths = _resolve_image_paths(args.images)
    else:
        if not args.image_dir.is_dir():
            raise SystemExit(f"Image directory not found: {args.image_dir}")
        paths = _list_images(args.image_dir)

    if args.limit is not None:
        paths = paths[: args.limit]

    done = _load_existing(args.out) if args.resume else {}
    pending = [p for p in paths if p.stem not in done]
    if args.max_images is not None:
        pending = pending[: args.max_images]

    print(
        f"images={len(paths)} already={len(done)} this_run={len(pending)} "
        f"quant={quant} adapter={adapter_dir} model_dir={args.model_dir}"
    )
    if not pending:
        ordered = [done[p.stem] for p in paths if p.stem in done]
        write_predictions(args.out, ordered)
        print(f"Nothing to do; wrote {args.out}")
        return

    print(f"Loading model (quant={quant}) …")
    gen = Qwen3VLLoRAGenerator(
        adapter_dir,
        model_dir=args.model_dir,
        max_new_tokens=args.max_new_tokens,
        image_max_pixels=args.image_max_pixels,
        quant=quant,  # type: ignore[arg-type]
    )
    allowed = set(closed_labels())

    import gc

    import torch

    new_count = 0
    for path in tqdm(pending, desc="infer_full_winner"):
        raw = gen.generate(path, image_id=path.stem)
        row = postprocess_prediction(raw, default_image_id=path.stem)
        row["damage_categories"] = [c for c in row["damage_categories"] if c in allowed]
        done[path.stem] = row
        new_count += 1
        if new_count % args.save_every == 0:
            write_predictions(args.out, [done[p.stem] for p in paths if p.stem in done])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ordered = [done[p.stem] for p in paths if p.stem in done]
    write_predictions(args.out, ordered)
    print(f"Wrote {len(ordered)}/{len(paths)} predictions to {args.out}")
    if len(ordered) < len(paths):
        print("Partial run. Re-run with --resume (optionally --max-images 20).")


if __name__ == "__main__":
    main()

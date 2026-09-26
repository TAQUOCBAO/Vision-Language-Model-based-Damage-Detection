#!/usr/bin/env python3
"""Run holdout (val) inference with a LoRA adapter and write predictions JSON.

Example:
  python scripts/infer_holdout.py --tier a --adapter outputs/runs/tier_a
  python scripts/04_eval_holdout.py --tier a --predictions-json outputs/preds/tier_a_val.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from damage_vlm.config import REPO_ROOT
from damage_vlm.infer.predict_folder import write_predictions
from damage_vlm.infer.qwen_vl import Qwen3VLLoRAGenerator
from damage_vlm.postprocess.pipeline import postprocess_prediction


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="a", help="Tier tag used in output filename")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA adapter dir (default: outputs/runs/tier_<tier>)",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "splits" / "v1.json",
    )
    parser.add_argument(
        "--silver-json",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "silver_labels.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Predictions JSON (default: outputs/preds/tier_<tier>_val.json)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip image_ids already present in --out",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Flush predictions to disk every N new images",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process at most this many pending images then exit (for chunked runs)",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    adapter = args.adapter or (REPO_ROOT / "outputs" / "runs" / f"tier_{args.tier}")
    out = args.out or (REPO_ROOT / "outputs" / "preds" / f"tier_{args.tier}_val.json")

    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"Adapter not found under {adapter}")

    silver = json.loads(args.silver_json.read_text(encoding="utf-8"))
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    val_ids = list(split["val_ids"])
    if args.limit is not None:
        val_ids = val_ids[: args.limit]

    done = _load_existing(out) if args.resume else {}
    pending = [i for i in val_ids if i not in done]
    if args.max_images is not None:
        pending = pending[: args.max_images]
    print(f"Val={len(val_ids)} already={len(done)} this_run={len(pending)}")

    if not pending:
        write_predictions(out, [done[i] for i in val_ids if i in done])
        print(f"Nothing to do; wrote {out}")
        return

    print(f"Loading adapter from {adapter} …")
    gen = Qwen3VLLoRAGenerator(
        adapter,
        max_new_tokens=args.max_new_tokens,
        image_max_pixels=args.image_max_pixels,
    )

    import gc

    import torch

    new_count = 0
    for image_id in tqdm(pending, desc=f"infer tier_{args.tier}"):
        if image_id not in silver:
            raise SystemExit(f"Missing silver entry for val id {image_id}")
        image_path = REPO_ROOT / silver[image_id]["image_path"]
        if not image_path.is_file():
            raise SystemExit(f"Missing image file: {image_path}")
        raw = gen.generate(image_path, image_id=image_id)
        done[image_id] = postprocess_prediction(raw, default_image_id=image_id)
        new_count += 1
        if new_count % args.save_every == 0:
            write_predictions(out, [done[i] for i in val_ids if i in done])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_predictions(out, [done[i] for i in val_ids if i in done])
    n_done = len([i for i in val_ids if i in done])
    print(f"Wrote {n_done}/{len(val_ids)} predictions to {out}")
    if n_done >= len(val_ids):
        print(
            "Next: "
            f"python scripts/04_eval_holdout.py --tier {args.tier} --predictions-json {out}"
        )
    else:
        print(f"Partial run complete ({n_done}/{len(val_ids)}). Re-run with --resume.")


if __name__ == "__main__":
    main()

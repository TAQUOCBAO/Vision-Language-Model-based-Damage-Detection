#!/usr/bin/env python3
"""Step 10 — Evaluate F1 / METEOR / weighted score on a chosen image set.

Ground truth comes from silver labels (default: data_v2 full-winner silver).
Predictions are either:
  - generated from a local model path (``--adapter`` or ``--model-dir``), or
  - loaded from an existing ``--predictions-json``.

Point ``--adapter`` / ``--model-dir`` at the folder that holds the fine-tuned
weights (same paths as Step 8). Prefer the full-precision LoRA under
``outputs/runs/full_winner_v2`` when VRAM allows; use the low-VRAM export under
``outputs/exports/full_winner_v2_lora`` with ``--load-in-4bit`` only if needed.

Examples:
  # Full-precision LoRA (recommended)
  python scripts/11_eval_metrics.py \\
    --adapter outputs/runs/full_winner_v2 \\
    --image-dir data_v2/dataset/image --limit 20

  # Low-VRAM package
  python scripts/11_eval_metrics.py --load-in-4bit \\
    --adapter outputs/exports/full_winner_v2_lora \\
    --image data_v2/dataset/image/00002.jpg

  # Merged model folder
  python scripts/11_eval_metrics.py --load-in-4bit \\
    --model-dir outputs/exports/full_winner_v2_lora_merged \\
    --image path/to.jpg

  # Metrics only (reuse predictions; no model load)
  python scripts/11_eval_metrics.py \\
    --predictions-json outputs/submissions/submission.json \\
    --image-dir data_v2/dataset/image
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from damage_vlm.config import REPO_ROOT, closed_labels
from damage_vlm.infer.predict_folder import write_predictions
from damage_vlm.infer.qwen_vl import Qwen3VLLoRAGenerator
from damage_vlm.metrics.f1 import multilabel_f1
from damage_vlm.metrics.meteor_score import meteor_mean
from damage_vlm.postprocess.pipeline import postprocess_prediction
from damage_vlm.select.winner import weighted_score

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


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


def _load_preds(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"Predictions must be a JSON list: {path}")
    return {r["image_id"]: r for r in rows if isinstance(r, dict) and "image_id" in r}


def _run_infer(
    paths: list[Path],
    *,
    adapter: Path | None,
    model_dir: Path | None,
    quant: str,
    max_new_tokens: int,
    image_max_pixels: int,
) -> dict[str, dict]:
    gen = Qwen3VLLoRAGenerator(
        adapter,
        model_dir=model_dir,
        max_new_tokens=max_new_tokens,
        image_max_pixels=image_max_pixels,
        quant=quant,  # type: ignore[arg-type]
    )
    allowed = set(closed_labels())
    preds: dict[str, dict] = {}
    for path in tqdm(paths, desc="infer_for_eval"):
        raw = gen.generate(path, image_id=path.stem)
        row = postprocess_prediction(raw, default_image_id=path.stem)
        row["damage_categories"] = [c for c in row["damage_categories"] if c in allowed]
        preds[path.stem] = row
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        "--image-path",
        dest="images",
        type=Path,
        action="append",
        default=None,
        help="Image file to evaluate (repeatable). Overrides --image-dir when set.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory of images to evaluate (default: data_v2/dataset/image if no --image)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--predictions-json",
        type=Path,
        default=None,
        help="Reuse predictions; skip model inference",
    )
    parser.add_argument(
        "--silver-json",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "full_winner" / "silver_labels.json",
        help="Gold labels JSON (default: full_winner / data_v2 silver)",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPO_ROOT / "outputs" / "runs" / "full_winner_v2",
        help="Path to LoRA adapter directory (recommended: outputs/runs/full_winner_v2)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to a merged full model directory (skips --adapter)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load backbone in bitsandbytes NF4 (use with low-VRAM export path)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load backbone in bitsandbytes 8-bit",
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument(
        "--preds-out",
        type=Path,
        default=None,
        help="Where to write predictions used for this eval "
        "(default: outputs/preds/eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Metrics report JSON (default: outputs/evals/eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--tag",
        default="custom",
        help="Label stored in the report (default: custom)",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Choose only one of --load-in-4bit / --load-in-8bit")
    quant = "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else "none"

    if args.images:
        paths = _resolve_image_paths(args.images)
    else:
        image_dir = args.image_dir or (REPO_ROOT / "data_v2" / "dataset" / "image")
        if not image_dir.is_dir():
            raise SystemExit(f"Image directory not found: {image_dir}")
        paths = _list_images(image_dir)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("No images selected for evaluation")

    if not args.silver_json.is_file():
        fallback = REPO_ROOT / "data" / "processed" / "silver_labels.json"
        if fallback.is_file():
            print(f"WARNING: {args.silver_json} missing; using {fallback}")
            args.silver_json = fallback
        else:
            raise SystemExit(
                f"Silver labels not found: {args.silver_json}\n"
                "Build with: python scripts/07_build_full_winner.py --dataset-root data_v2/dataset"
            )
    silver = json.loads(args.silver_json.read_text(encoding="utf-8"))

    eval_ids = [p.stem for p in paths]
    missing_gold = [i for i in eval_ids if i not in silver]
    if missing_gold:
        raise SystemExit(
            f"No silver labels for {len(missing_gold)} image(s), e.g. {missing_gold[:5]}. "
            "Use images from the corpus covered by --silver-json."
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    preds_out = args.preds_out or (REPO_ROOT / "outputs" / "preds" / f"eval_{stamp}.json")
    metrics_out = args.out or (REPO_ROOT / "outputs" / "evals" / f"eval_{stamp}.json")

    if args.predictions_json is not None:
        if not args.predictions_json.is_file():
            raise SystemExit(f"Predictions not found: {args.predictions_json}")
        all_preds = _load_preds(args.predictions_json)
        missing_pred = [i for i in eval_ids if i not in all_preds]
        if missing_pred:
            raise SystemExit(
                f"Missing predictions for {len(missing_pred)} id(s), e.g. {missing_pred[:5]}"
            )
        preds = {i: all_preds[i] for i in eval_ids}
        print(f"Loaded predictions from {args.predictions_json}")
    else:
        if args.model_dir is not None:
            if not args.model_dir.is_dir():
                raise SystemExit(f"Model directory not found: {args.model_dir}")
            adapter_for_load: Path | None = None
        else:
            if not (args.adapter / "adapter_config.json").is_file():
                raise SystemExit(
                    f"Adapter not found under {args.adapter}\n"
                    "Pass --adapter PATH (e.g. outputs/runs/full_winner_v2) "
                    "or --model-dir PATH, or --predictions-json."
                )
            adapter_for_load = args.adapter
        print(
            f"Running inference (quant={quant}) on {len(paths)} image(s) "
            f"adapter={adapter_for_load} model_dir={args.model_dir} …"
        )
        preds = _run_infer(
            paths,
            adapter=adapter_for_load,
            model_dir=args.model_dir,
            quant=quant,
            max_new_tokens=args.max_new_tokens,
            image_max_pixels=args.image_max_pixels,
        )

    write_predictions(preds_out, [preds[i] for i in eval_ids])

    y_true = [silver[i]["damage_categories"] for i in eval_ids]
    y_pred = [preds[i]["damage_categories"] for i in eval_ids]
    refs = [silver[i]["description"] for i in eval_ids]
    hyps = [preds[i].get("description", "") for i in eval_ids]

    f1 = multilabel_f1(y_true, y_pred)
    try:
        meteor = meteor_mean(refs, hyps)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: METEOR unavailable ({exc}); using 0.0")
        meteor = 0.0

    report = {
        "tag": args.tag,
        "n": len(eval_ids),
        "f1": f1,
        "meteor": meteor,
        "weighted_score": weighted_score(f1, meteor),
        "silver_json": str(args.silver_json),
        "predictions_json": str(preds_out),
        "quant": quant,
        "adapter": str(args.adapter) if args.model_dir is None else None,
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "image_ids": eval_ids,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "image_ids"}
    summary["n_image_ids"] = len(eval_ids)
    print(json.dumps(summary, indent=2))
    print(f"Metrics report: {metrics_out}")
    print(f"Predictions:    {preds_out}")


if __name__ == "__main__":
    main()

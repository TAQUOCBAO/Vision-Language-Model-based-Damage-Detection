#!/usr/bin/env python3
"""Official hidden-test runner (Test Requirements.docx).

Uses the organizer prompt template verbatim:

  Q1: Determine whether there is structural damage in the image?
  Q2: Describe the damage characteristics based on the image?

Then writes JSON predictions for 001.jpg–110.jpg. If a gold file is provided,
scores category accuracy and METEOR as specified by the organizers.

Examples:
  python scripts/12_test_official.py
  python scripts/12_test_official.py --load-in-4bit --limit 2
  python scripts/12_test_official.py --predictions-json outputs/submissions/official_test.json
  python scripts/12_test_official.py --gold-json path/to/official_gold.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from damage_vlm.config import REPO_ROOT, closed_labels, load_prompts_config
from damage_vlm.data.official_test import (
    DEFAULT_OFFICIAL_IMAGE_DIR,
    DEFAULT_OFFICIAL_REQUIREMENTS,
    OFFICIAL_TEST_COUNT,
    list_official_test_images,
    load_official_gold,
)
from damage_vlm.infer.predict_folder import write_predictions
from damage_vlm.metrics.official import official_scores
from damage_vlm.postprocess.pipeline import postprocess_prediction

PROMPT_DOC = (
    "The dataset contains 110 structural damage images, named 001.jpg to 110.jpg. "
    "Participants are required to use the Specified Prompt Template to enable the "
    "model to accomplish the following tasks: (1) determine the damage categories "
    "present in the image; (2) provide damage characteristic descriptions, including "
    "fine-grained information such as crack orientation, corrosion severity, void size."
)

# bf16 8B + LoRA + vision activations; 4/8-bit leave headroom for generate().
_MIN_FREE_GIB = {"none": 20.0, "8bit": 12.0, "4bit": 8.0}


def _gpu_compute_apps() -> str:
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv",
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.strip()


def _require_gpu_memory(quant: str) -> None:
    import torch

    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb = free_b / (1024**3)
    total_gb = total_b / (1024**3)
    need = _MIN_FREE_GIB.get(quant, 20.0)
    print(f"GPU memory: {free_gb:.2f} GiB free / {total_gb:.2f} GiB total  (need ≥ {need:.0f} GiB for quant={quant})")
    if free_gb >= need:
        return
    apps = _gpu_compute_apps()
    extra = f"\nOther CUDA processes:\n{apps}" if apps else ""
    raise SystemExit(
        f"Not enough free VRAM to load Qwen3-VL-8B (quant={quant}). "
        f"Free {free_gb:.2f} GiB, need about {need:.0f} GiB.{extra}\n"
        "A text-generation-server (google/gemma-2-9b-it) on this machine "
        "auto-restarts and takes ~42 GiB. Stop/disable that service, then --resume."
    )


def _pin_gpu(leave_gib: float = 2.0):
    """Hold almost all free VRAM so a restarting TGI job cannot grab the GPU."""
    import torch

    torch.cuda.set_device(0)
    free_b, _ = torch.cuda.mem_get_info(0)
    take = int(free_b - leave_gib * (1024**3))
    need = int(_MIN_FREE_GIB["none"] * (1024**3))
    if take < need:
        raise SystemExit(
            f"Cannot pin GPU: only {free_b / (1024**3):.2f} GiB free.\n"
            f"{_gpu_compute_apps()}"
        )
    print(f"Pinning {take / (1024**3):.1f} GiB on cuda:0 until the model is placed")
    return torch.empty(take // 2, dtype=torch.float16, device="cuda:0")


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


def _finalize_row(raw: str, image_id: str) -> dict:
    allowed = set(closed_labels())
    row = postprocess_prediction(raw, default_image_id=image_id)
    cats = [c for c in row["damage_categories"] if c in allowed]
    has_damage = bool(cats)
    return {
        "image_id": image_id,
        "q1_has_structural_damage": has_damage,
        "damage_categories": cats,
        "description": row["description"],
    }


def _run_infer(
    paths: list[Path],
    *,
    adapter: Path | None,
    model_dir: Path | None,
    quant: str,
    max_new_tokens: int,
    image_max_pixels: int,
    out: Path,
    resume: bool,
    save_every: int,
    device: str = "cuda:0",
    gpu_hold: list | None = None,
) -> dict[str, dict]:
    done = _load_existing(out) if resume else {}
    pending = [p for p in paths if p.stem not in done]
    print(
        f"official_test images={len(paths)} already={len(done)} this_run={len(pending)} "
        f"quant={quant} device={device} adapter={adapter} model_dir={model_dir}"
    )
    if not pending:
        return {p.stem: done[p.stem] for p in paths if p.stem in done}

    print(f"Loading model on {device} (quant={quant}, official Q1/Q2 prompt) …")
    import torch
    from damage_vlm.infer.qwen_vl import Qwen3VLLoRAGenerator

    gen = Qwen3VLLoRAGenerator(
        adapter,
        model_dir=model_dir,
        device=device,
        max_new_tokens=max_new_tokens,
        image_max_pixels=image_max_pixels,
        quant=quant,  # type: ignore[arg-type]
        gpu_hold=gpu_hold,
    )
    param_dev = next(gen.model.parameters()).device
    print(f"Model parameters on {param_dev}")
    if str(device).startswith("cuda"):
        if param_dev.type != "cuda":
            raise SystemExit(f"Model is on {param_dev}, expected GPU ({device}).")
        print(
            f"GPU memory allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB  "
            f"reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )

    import gc

    new_count = 0
    for path in tqdm(pending, desc="official_test"):
        raw = gen.generate(path, image_id=path.stem, official=True)
        done[path.stem] = _finalize_row(raw, path.stem)
        new_count += 1
        if new_count % save_every == 0:
            write_predictions(out, [done[p.stem] for p in paths if p.stem in done])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {p.stem: done[p.stem] for p in paths if p.stem in done}


def _prediction_stats(rows: list[dict]) -> dict:
    cat_counts: Counter[str] = Counter()
    for row in rows:
        cat_counts.update(row.get("damage_categories") or [])
    n_damage = sum(1 for r in rows if r.get("q1_has_structural_damage"))
    n_empty_desc = sum(1 for r in rows if not str(r.get("description") or "").strip())
    return {
        "n_predictions": len(rows),
        "n_q1_has_damage": n_damage,
        "n_q1_no_damage": len(rows) - n_damage,
        "n_empty_description": n_empty_desc,
        "mean_categories_per_image": (
            sum(len(r.get("damage_categories") or []) for r in rows) / len(rows) if rows else 0.0
        ),
        "category_histogram": dict(cat_counts),
    }


def _write_markdown_report(
    path: Path,
    *,
    image_dir: Path,
    preds_path: Path,
    rows: list[dict],
    stats: dict,
    scores: dict | None,
    gold_path: Path | None,
    quant: str,
    adapter: Path | None,
    model_dir: Path | None,
    limit: int | None,
) -> None:
    prompts = load_prompts_config()
    q1 = str(prompts["official_q1"]).strip()
    q2 = str(prompts["official_q2"]).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    samples = rows[:3]
    sample_block = json.dumps(samples, indent=2, ensure_ascii=False) if samples else "[]"
    hist_lines = "\n".join(
        f"| `{name}` | {count} |"
        for name, count in sorted(stats.get("category_histogram", {}).items(), key=lambda kv: (-kv[1], kv[0]))
    ) or "| *(none)* | 0 |"

    if scores:
        score_block = f"""## Official scores (gold labels provided)

| Axis | Metric | Value |
|------|--------|-------|
| Q1 presence | Binary accuracy | {scores['q1_binary_accuracy']:.6f} |
| Q1 categories | Subset accuracy (exact set match) | {scores['category_subset_accuracy']:.6f} |
| Q1 categories | Label accuracy (1 − Hamming loss) | {scores['category_label_accuracy']:.6f} |
| Q1 categories | Sample-average F1 (supplementary) | {scores['category_sample_f1']:.6f} |
| Q2 description | METEOR | {scores['meteor']:.6f} |

Gold file: `{gold_path}`
"""
    else:
        score_block = """## Official scores

The public test package does **not** include hidden ground-truth labels.
Category accuracy and METEOR can be computed only after the organizer gold
file is supplied:

```bash
python scripts/12_test_official.py \\
  --predictions-json outputs/submissions/official_test.json \\
  --gold-json path/to/official_gold.json
```
"""

    body = f"""# Official Test Report — Structural Damage Image–Text VLM

**Generated:** {stamp}  
**Specification:** `Test Requirements.docx` (110 images, official Q1/Q2 prompt, category accuracy + METEOR)

## Compliance checklist

| Requirement | Status |
|-------------|--------|
| 110 images named `001.jpg`–`110.jpg` | Yes (`{image_dir}`) |
| Specified prompt Q1 used verbatim | `{q1}` |
| Specified prompt Q2 used verbatim | `{q2}` |
| Task 1: damage category classification | `damage_categories` from closed 10-class set |
| Task 2: fine-grained characteristic description | `description` (orientation / severity / size) |
| Evaluation axis 1: category accuracy vs gold | Implemented (`scripts/12_test_official.py --gold-json`) |
| Evaluation axis 2: METEOR vs gold descriptions | Implemented (NLTK METEOR) |
| Reproducible offline code | This script + `src/damage_vlm/` |

{PROMPT_DOC}

## Model under test

| Field | Value |
|-------|-------|
| Adapter | `{adapter}` |
| Merged model | `{model_dir}` |
| Quantization | `{quant}` |
| Predictions | `{preds_path}` |
| Images scored | {stats['n_predictions']}{' (limit applied)' if limit else f' / {OFFICIAL_TEST_COUNT}'} |

## Prediction summary (this run)

| Statistic | Value |
|-----------|-------|
| Q1 = yes (has damage) | {stats['n_q1_has_damage']} |
| Q1 = no (no closed-set damage) | {stats['n_q1_no_damage']} |
| Empty descriptions | {stats['n_empty_description']} |
| Mean categories / image | {stats['mean_categories_per_image']:.3f} |

### Predicted category histogram

| Category | Count |
|----------|-------|
{hist_lines}

{score_block}

## Sample predictions

```json
{sample_block}
```

## How this maps onto the organizer questions

- **Q1** is answered as `q1_has_structural_damage` (true iff at least one closed-set category is predicted) plus the multi-label `damage_categories` list.
- **Q2** is answered as the technical `description` after the shared post-process stack (JSON repair, synonym map, category–text alignment, sentence de-duplication).

## Reproducibility

```bash
python scripts/12_test_official.py --adapter outputs/runs/full_winner_v2
# Low VRAM:
python scripts/12_test_official.py --load-in-4bit --adapter outputs/runs/full_winner_v2
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_OFFICIAL_IMAGE_DIR,
        help="Official test image folder (001.jpg–110.jpg)",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPO_ROOT / "outputs" / "runs" / "full_winner_v2",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback if CUDA is down (very slow; default is GPU-only)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test first N official images")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument(
        "--predictions-json",
        type=Path,
        default=None,
        help="Reuse existing official predictions; skip model load",
    )
    parser.add_argument(
        "--gold-json",
        type=Path,
        default=None,
        help="Organizer gold labels (hidden; optional)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "submissions" / "official_test.json",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=REPO_ROOT / "outputs" / "evals" / "official_test.json",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=REPO_ROOT / "OFFICIAL_TEST_REPORT.md",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Choose only one of --load-in-4bit / --load-in-8bit")
    quant = "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else "none"

    if args.predictions_json is None and not args.allow_cpu:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is required for official test "
                f"(torch {torch.__version__}, cuda_built={torch.version.cuda}). "
                "torch.cuda.is_available() is False. Fix the GPU/driver, then re-run. "
                "Pass --allow-cpu only if you intentionally want CPU."
            )
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  "
            f"(device=cuda:0, quant={quant}, torch={torch.__version__})"
        )
        _require_gpu_memory(quant)

    gpu_hold: list | None = None
    if args.predictions_json is None and not args.allow_cpu and quant == "none":
        gpu_hold = [_pin_gpu()]

    paths = list_official_test_images(args.image_dir)
    if args.limit is not None:
        paths = paths[: args.limit]
    eval_ids = [p.stem for p in paths]

    prompts = load_prompts_config()
    print(str(prompts["official_q1"]).strip())
    print(str(prompts["official_q2"]).strip())
    print(f"requirements_doc={DEFAULT_OFFICIAL_REQUIREMENTS}")
    print(f"image_dir={args.image_dir} n={len(paths)}")

    if args.predictions_json is not None:
        if not args.predictions_json.is_file():
            raise SystemExit(f"Predictions not found: {args.predictions_json}")
        loaded = _load_existing(args.predictions_json)
        missing = [i for i in eval_ids if i not in loaded]
        if missing:
            raise SystemExit(f"Missing predictions for {len(missing)} id(s), e.g. {missing[:5]}")
        preds = {i: loaded[i] for i in eval_ids}
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
                    "Train: python scripts/03_train_tier.py --tier full\n"
                    "Or download the Hub LoRA and pass --adapter PATH"
                )
            adapter_for_load = args.adapter
        infer_device = "cuda:0"
        if args.allow_cpu:
            import torch

            infer_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        preds = _run_infer(
            paths,
            adapter=adapter_for_load,
            model_dir=args.model_dir,
            quant=quant,
            max_new_tokens=args.max_new_tokens,
            image_max_pixels=args.image_max_pixels,
            out=args.out,
            resume=args.resume,
            save_every=args.save_every,
            device=infer_device,
            gpu_hold=gpu_hold,
        )
        missing = [i for i in eval_ids if i not in preds]
        if missing:
            raise SystemExit(
                f"Incomplete run ({len(preds)}/{len(eval_ids)}). Re-run with --resume."
            )

    rows = [preds[i] for i in eval_ids]
    write_predictions(args.out, rows)
    stats = _prediction_stats(rows)

    scores = None
    if args.gold_json is not None:
        if not args.gold_json.is_file():
            raise SystemExit(f"Gold labels not found: {args.gold_json}")
        gold = load_official_gold(args.gold_json)
        missing_gold = [i for i in eval_ids if i not in gold]
        if missing_gold:
            raise SystemExit(
                f"Gold file missing {len(missing_gold)} official id(s), e.g. {missing_gold[:5]}"
            )
        scores = official_scores(
            [gold[i]["damage_categories"] for i in eval_ids],
            [preds[i]["damage_categories"] for i in eval_ids],
            [gold[i]["description"] for i in eval_ids],
            [preds[i].get("description", "") for i in eval_ids],
            true_has_damage=[gold[i]["has_structural_damage"] for i in eval_ids],
            pred_has_damage=[bool(preds[i].get("q1_has_structural_damage")) for i in eval_ids],
        )

    report = {
        "tag": "official_test",
        "spec": "Test Requirements.docx",
        "n": len(eval_ids),
        "image_dir": str(args.image_dir),
        "predictions_json": str(args.out),
        "quant": quant,
        "adapter": str(args.adapter) if args.model_dir is None else None,
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "official_q1": str(prompts["official_q1"]).strip(),
        "official_q2": str(prompts["official_q2"]).strip(),
        "prediction_stats": stats,
        "scores": scores,
        "gold_json": str(args.gold_json) if args.gold_json else None,
        "image_ids": eval_ids,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _write_markdown_report(
        args.report_out,
        image_dir=args.image_dir,
        preds_path=args.out,
        rows=rows,
        stats=stats,
        scores=scores,
        gold_path=args.gold_json,
        quant=quant,
        adapter=args.adapter if args.model_dir is None else None,
        model_dir=args.model_dir,
        limit=args.limit,
    )

    printable = {k: v for k, v in report.items() if k != "image_ids"}
    printable["n_image_ids"] = len(eval_ids)
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"Predictions: {args.out}")
    print(f"Metrics:     {args.metrics_out}")
    print(f"Report:      {args.report_out}")


if __name__ == "__main__":
    main()

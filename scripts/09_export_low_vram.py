#!/usr/bin/env python3
"""Export a low-VRAM deploy package from a full-winner LoRA adapter.

Creates:
  1) ``outputs/exports/<name>/`` — LoRA adapter + ``deploy.json`` (4-bit infer recipe)
  2) ``outputs/exports/<name>.tar.gz`` — compressed archive of that folder (~0.7 GB)

VRAM savings come from loading the 8B base in **bitsandbytes 4-bit** at inference
time (see ``scripts/08_infer_full_winner.py --load-in-4bit``), not from gzip alone.

Optional ``--merge`` also writes a merged full model directory (large, ~16 GB on disk)
for single-folder deploy; prefer the LoRA package + ``--load-in-4bit`` for low VRAM.

Examples:
  python scripts/09_export_low_vram.py
  python scripts/09_export_low_vram.py --adapter outputs/runs/full_winner_v2
  # Then infer:
  python scripts/08_infer_full_winner.py --load-in-4bit \\
    --adapter outputs/exports/full_winner_v2_lora \\
    --image data_v2/dataset/image/00002.jpg
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from damage_vlm.config import REPO_ROOT
from damage_vlm.infer.qwen_vl import DEFAULT_BASE_MODEL

ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "processor_config.json",
    "chat_template.jinja",
    "README.md",
)


def _copy_adapter(src: Path, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ADAPTER_FILES:
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)
            copied.append(name)
    if "adapter_config.json" not in copied or "adapter_model.safetensors" not in copied:
        raise SystemExit(f"Incomplete adapter under {src} (need adapter_*. files)")
    return copied


def _make_tar_gz(folder: Path, archive: Path) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)
    return archive


def _merge_and_save(adapter: Path, out_dir: Path, base_model: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Merging LoRA into {base_model} (needs GPU/CPU RAM) …")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out_dir} …")
    model.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    (out_dir / "export_meta.json").write_text(
        json.dumps(
            {
                "kind": "merged_full_model",
                "base_model": base_model,
                "adapter": str(adapter),
                "note": "Load with --model-dir and optional --load-in-4bit for low VRAM",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPO_ROOT / "outputs" / "runs" / "full_winner_v2",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "exports" / "full_winner_v2_lora",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Output .tar.gz path (default: <out-dir>.tar.gz)",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Hub id still needed at 4-bit infer time (base weights)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Also write a merged full model under <out-dir>_merged (large on disk)",
    )
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()

    if not (args.adapter / "adapter_config.json").is_file():
        raise SystemExit(f"Adapter not found: {args.adapter}")

    copied = _copy_adapter(args.adapter, args.out_dir)
    deploy = {
        "kind": "lora_low_vram_package",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter_dir": str(args.out_dir.relative_to(REPO_ROOT))
        if args.out_dir.is_relative_to(REPO_ROOT)
        else str(args.out_dir),
        "recommended_infer": {
            "load_in_4bit": True,
            "image_max_pixels": 262144,
            "max_new_tokens": 384,
            "command": (
                "python scripts/08_infer_full_winner.py --load-in-4bit "
                f"--adapter {args.out_dir} --image path/to.jpg"
            ),
        },
        "files": copied,
        "notes": [
            "Archive contains LoRA weights only (~0.7 GB).",
            "At inference, the 8B base is loaded in NF4 (bitsandbytes) to cut VRAM.",
            "Install: pip install bitsandbytes qwen-vl-utils",
        ],
    }
    (args.out_dir / "deploy.json").write_text(
        json.dumps(deploy, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote package dir: {args.out_dir}")
    print(json.dumps(deploy["recommended_infer"], indent=2))

    archive = args.archive or Path(str(args.out_dir) + ".tar.gz")
    if not args.skip_archive:
        _make_tar_gz(args.out_dir, archive)
        size_gb = archive.stat().st_size / (1024**3)
        print(f"Wrote archive: {archive} ({size_gb:.2f} GiB)")

    if args.merge:
        merged = Path(str(args.out_dir) + "_merged")
        _merge_and_save(args.adapter, merged, args.base_model)
        print(f"Merged model: {merged}")
        print(
            "Low-VRAM infer from merged:\n"
            f"  python scripts/08_infer_full_winner.py --load-in-4bit "
            f"--model-dir {merged} --image path/to.jpg"
        )


if __name__ == "__main__":
    main()

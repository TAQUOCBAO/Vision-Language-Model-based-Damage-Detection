#!/usr/bin/env python3
"""Push a fine-tuned LoRA (or low-VRAM export package) to the Hugging Face Hub.

Prerequisites:
  pip install -U "huggingface_hub"
  huggingface-cli login          # or set HF_TOKEN

Examples:
  # Full-winner LoRA (recommended artifact)
  python scripts/10_push_to_hub.py --repo-id YOUR_USER/damage-vlm-qwen3vl-8b-lora

  # Dry-run (list files only)
  python scripts/10_push_to_hub.py --repo-id YOUR_USER/damage-vlm-lora --dry-run

  # Private repo + low-VRAM export package
  python scripts/10_push_to_hub.py \\
    --repo-id YOUR_USER/damage-vlm-lora-4bit-pkg \\
    --model-dir outputs/exports/full_winner_v2_lora \\
    --private

  # Also upload the .tar.gz next to the folder
  python scripts/10_push_to_hub.py \\
    --repo-id YOUR_USER/damage-vlm-lora \\
    --include-archive
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from damage_vlm.config import REPO_ROOT
from damage_vlm.infer.qwen_vl import DEFAULT_BASE_MODEL

# Training / export junk we never want on the Hub by default.
DEFAULT_IGNORE = [
    "checkpoint-*/**",
    "checkpoint-*",
    "*.bin",
    "training_args.bin",
    "trainer_state.json",
    "trainer_log.jsonl",
    "train_results.json",
    "all_results.json",
    "training_loss.png",
    "hf_logs/**",
    "runs/**",
]


def _require_hub():
    try:
        from huggingface_hub import HfApi, create_repo, whoami
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            'Install Hub client: pip install -U "huggingface_hub"\n'
            "Then: huggingface-cli login"
        ) from exc
    return HfApi, create_repo, whoami


def _write_model_card(
    dest: Path,
    *,
    repo_id: str,
    base_model: str,
    model_dir: Path,
    kind: str,
) -> Path:
    card = f"""---
library_name: peft
base_model: {base_model}
tags:
  - qwen3-vl
  - lora
  - structural-damage
  - image-text-to-text
license: apache-2.0
pipeline_tag: image-text-to-text
---

# {repo_id.split("/")[-1]}

LoRA adapter for **{base_model}** fine-tuned for structural damage diagnosis
(`image_id`, `damage_categories`, `description` JSON).

- **Source dir:** `{model_dir}`
- **Package kind:** `{kind}`
- **Exported (UTC):** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")}

## Load (Python)

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = "{base_model}"
adapter = "{repo_id}"

processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter)
```

## Low-VRAM (4-bit)

```bash
pip install bitsandbytes
# In this repo:
python scripts/08_infer_full_winner.py --load-in-4bit \\
  --adapter {repo_id} --image path/to.jpg
```

## Project

Offline competition pipeline — see the training repository README for data tiers,
holdout bake-off, and full-winner retrain steps.
"""
    path = dest / "README.md"
    # Prefer generating a Hub card; keep existing only if --keep-readme
    path.write_text(card, encoding="utf-8")
    return path


def _detect_kind(model_dir: Path) -> str:
    if (model_dir / "deploy.json").is_file():
        return "low_vram_lora_package"
    if (model_dir / "adapter_config.json").is_file():
        return "lora_adapter"
    if (model_dir / "config.json").is_file():
        return "merged_or_full_model"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hub repo id, e.g. username/damage-vlm-qwen3vl-8b-lora",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "runs" / "full_winner_v2",
        help="Local fine-tuned dir to upload (LoRA run or exports/…_lora)",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model id written into the model card",
    )
    parser.add_argument("--private", action="store_true", help="Create/update as private")
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision / branch on the Hub (default: main)",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message (default: auto)",
    )
    parser.add_argument(
        "--keep-readme",
        action="store_true",
        help="Do not overwrite local README.md with a generated model card",
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Also upload <model-dir>.tar.gz if it exists next to the folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be uploaded; do not call the Hub",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (default: HF_TOKEN env or cached login)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.is_absolute():
        model_dir = (REPO_ROOT / model_dir).resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")

    kind = _detect_kind(model_dir)
    if kind == "unknown":
        raise SystemExit(
            f"No adapter_config.json / deploy.json / config.json under {model_dir}"
        )

    token = args.token or os.environ.get("HF_TOKEN")
    card_path = None
    if not args.keep_readme:
        card_path = _write_model_card(
            model_dir,
            repo_id=args.repo_id,
            base_model=args.base_model,
            model_dir=model_dir.relative_to(REPO_ROOT)
            if model_dir.is_relative_to(REPO_ROOT)
            else model_dir,
            kind=kind,
        )

    # Manifest for local audit
    files = sorted(
        p.relative_to(model_dir).as_posix()
        for p in model_dir.rglob("*")
        if p.is_file()
    )
    # Filter obvious checkpoint trees from the listing message
    visible = [f for f in files if not f.startswith("checkpoint-")]
    print(f"Local dir: {model_dir}")
    print(f"Kind: {kind}")
    print(f"Files to consider ({len(visible)} non-checkpoint):")
    for f in visible[:40]:
        print(f"  {f}")
    if len(visible) > 40:
        print(f"  … +{len(visible) - 40} more")

    archive = Path(str(model_dir) + ".tar.gz")
    extra_files: list[Path] = []
    if args.include_archive:
        if archive.is_file():
            extra_files.append(archive)
            print(f"Will also upload archive: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"WARNING: --include-archive set but missing {archive}")

    if args.dry_run:
        print("Dry-run only; no upload.")
        return

    HfApi, create_repo, whoami = _require_hub()
    api = HfApi(token=token)
    try:
        user = whoami(token=token)
        print(f"Logged in as: {user.get('name') or user.get('fullname')}")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Hugging Face auth failed ({exc}).\n"
            "Run: huggingface-cli login   or export HF_TOKEN=…"
        ) from exc

    create_repo(
        args.repo_id,
        private=args.private,
        exist_ok=True,
        repo_type="model",
        token=token,
    )

    commit = args.commit_message or (
        f"Upload {kind} from {model_dir.name} ({datetime.now(timezone.utc).date()})"
    )
    print(f"Uploading folder → https://huggingface.co/{args.repo_id}")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message=commit,
        ignore_patterns=DEFAULT_IGNORE,
        token=token,
    )

    for extra in extra_files:
        print(f"Uploading {extra.name} …")
        api.upload_file(
            path_or_fileobj=str(extra),
            path_in_repo=extra.name,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            commit_message=f"Add {extra.name}",
            token=token,
        )

    meta = {
        "repo_id": args.repo_id,
        "url": f"https://huggingface.co/{args.repo_id}",
        "model_dir": str(model_dir),
        "kind": kind,
        "private": args.private,
        "card": str(card_path) if card_path else None,
        "pushed_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = REPO_ROOT / "outputs" / "exports" / "last_hub_push.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Done: {meta['url']}")


if __name__ == "__main__":
    main()

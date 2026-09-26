"""Validate LLaMA-Factory YAML configs and build train commands."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from damage_vlm.config import REPO_ROOT

DATASET_INFO_PATH = REPO_ROOT / "data" / "dataset_info.json"

REQUIRED_KEYS = {
    "model_name_or_path",
    "finetuning_type",
    "lora_rank",
    "lora_alpha",
    "lora_target",
    "freeze_vision_tower",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "image_max_pixels",
    "bf16",
    "learning_rate",
}

INSTALL_HINT = (
    "LLaMA-Factory is not installed (llamafactory-cli not found).\n"
    "Install into the active env, then retry:\n"
    '  pip install -U "llamafactory[torch,metrics]"\n'
    "Or from source:\n"
    "  pip install -U git+https://github.com/hiyouga/LLaMA-Factory.git\n"
    "Verify with:  which llamafactory-cli && llamafactory-cli version"
)


class LlamaFactoryNotFoundError(FileNotFoundError):
    """Raised when the LLaMA-Factory CLI cannot be located."""


def resolve_llamafactory_cli() -> list[str]:
    """Return argv prefix that launches LLaMA-Factory train CLI.

    Prefers ``llamafactory-cli`` on PATH, then ``python -m llamafactory.cli``.
    """
    cli = shutil.which("llamafactory-cli")
    if cli:
        return [cli]

    try:
        import llamafactory  # noqa: F401
    except ImportError as exc:
        raise LlamaFactoryNotFoundError(INSTALL_HINT) from exc

    return [sys.executable, "-m", "llamafactory.cli"]


def load_train_yaml(path: Path | str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid train yaml: {path}")
    return data


def validate_r12_config(cfg: dict[str, Any]) -> list[str]:
    """Return list of validation errors (empty if OK)."""
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in cfg:
            errors.append(f"missing key: {key}")
    if cfg.get("model_name_or_path") != "Qwen/Qwen3-VL-8B-Instruct":
        errors.append("model_name_or_path must be Qwen/Qwen3-VL-8B-Instruct")
    if cfg.get("per_device_train_batch_size") != 1:
        errors.append("per_device_train_batch_size should be 1 for VRAM-safe default")
    if int(cfg.get("gradient_accumulation_steps", 0)) < 8:
        errors.append("gradient_accumulation_steps should be >= 8")
    if cfg.get("freeze_vision_tower") is not True:
        errors.append("freeze_vision_tower should be true by default")
    if cfg.get("bf16") is not True:
        errors.append("bf16 should be true")
    return errors


def build_llamafactory_command(config_path: Path | str, *, dry_run: bool = False) -> list[str]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = load_train_yaml(path)
    errors = validate_r12_config(cfg)
    if errors:
        raise ValueError("Invalid train config: " + "; ".join(errors))
    if dry_run:
        cli_prefix = [shutil.which("llamafactory-cli") or "llamafactory-cli"]
        return ["echo", "DRY_RUN", *cli_prefix, "train", str(path)]
    cli_prefix = resolve_llamafactory_cli()
    return [*cli_prefix, "train", str(path)]


def dataset_info_snippet() -> dict[str, Any]:
    """Return LLaMA-Factory dataset_info entries for processed tiers.

    ``file_name`` is relative to ``dataset_dir`` (``data/``), so paths are
    ``processed/tiers/<tier>/train.jsonl``.
    """
    tags = {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    }
    columns = {"messages": "messages", "images": "images"}
    entries: dict[str, Any] = {
        name: {
            "file_name": f"processed/tiers/{folder}/train.jsonl",
            "formatting": "sharegpt",
            "columns": columns,
            "tags": tags,
        }
        for name, folder in (
            ("damage_tier_a", "a"),
            ("damage_tier_b", "b"),
            ("damage_tier_c", "c"),
        )
    }
    # Full retrain on winning recipe (all images), usually built from data_v2 via
    # scripts/07_build_full_winner.py → data/processed/full_winner/train.jsonl
    entries["damage_full_winner"] = {
        "file_name": "processed/full_winner/train.jsonl",
        "formatting": "sharegpt",
        "columns": columns,
        "tags": tags,
    }
    return entries


def write_dataset_info(path: Path | str | None = None) -> Path:
    """Merge tier entries into ``data/dataset_info.json`` and return the path."""
    out = Path(path) if path is not None else DATASET_INFO_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if out.is_file():
        try:
            loaded = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {}
    existing.update(dataset_info_snippet())
    out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return out

"""Helpers for loading YAML configs from the repository configs/ directory."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


@lru_cache(maxsize=4)
def load_labels_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else CONFIGS_DIR / "labels_v1.yaml"
    return load_yaml(cfg_path)


@lru_cache(maxsize=4)
def load_eval_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else CONFIGS_DIR / "eval.yaml"
    return load_yaml(cfg_path)


@lru_cache(maxsize=4)
def load_prompts_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else CONFIGS_DIR / "prompts.yaml"
    return load_yaml(cfg_path)


def closed_labels(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_labels_config()
    labels = list(cfg["closed_labels"])
    if len(labels) != 10:
        raise ValueError(f"Expected 10 closed labels, got {len(labels)}")
    return labels


def prefix_seeds(cfg: dict[str, Any] | None = None) -> dict[str, list[str]]:
    cfg = cfg or load_labels_config()
    return {str(k): list(v) for k, v in cfg.get("prefix_seeds", {}).items()}


def synonyms_longest_first(cfg: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Return synonym lists sorted by descending phrase length within each label."""
    cfg = cfg or load_labels_config()
    out: dict[str, list[str]] = {}
    for label, phrases in cfg.get("synonyms", {}).items():
        ordered = sorted((str(p) for p in phrases), key=lambda s: (-len(s), s))
        out[str(label)] = ordered
    return out


def alignment_templates(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    cfg = cfg or load_labels_config()
    return {str(k): str(v) for k, v in cfg.get("alignment_templates", {}).items()}

"""Train config validation tests."""

from __future__ import annotations

import builtins

import pytest

from damage_vlm.config import REPO_ROOT
from damage_vlm.train.register_dataset import (
    LlamaFactoryNotFoundError,
    build_llamafactory_command,
    load_train_yaml,
    resolve_llamafactory_cli,
    validate_r12_config,
)


def test_tier_a_yaml_has_r12_keys() -> None:
    path = REPO_ROOT / "configs" / "llamafactory" / "tier_a_lora_sft.yaml"
    cfg = load_train_yaml(path)
    assert validate_r12_config(cfg) == []
    assert cfg["lora_rank"] == 64
    assert cfg["image_max_pixels"] == 524288
    assert cfg["freeze_vision_tower"] is True


def test_dry_run_command() -> None:
    path = REPO_ROOT / "configs" / "llamafactory" / "tier_a_lora_sft.yaml"
    cmd = build_llamafactory_command(path, dry_run=True)
    assert cmd[0] == "echo"
    assert "train" in cmd
    assert any("llamafactory" in str(part) for part in cmd)


def test_resolve_cli_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import damage_vlm.train.register_dataset as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "llamafactory" or name.startswith("llamafactory."):
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(LlamaFactoryNotFoundError):
        resolve_llamafactory_cli()

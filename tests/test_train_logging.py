"""Tests for training log path helpers and dry-run log creation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from damage_vlm.train.logging_utils import make_train_log_path, write_log_header


def test_make_train_log_path_uses_utc_stamp(tmp_path: Path) -> None:
    when = datetime(2026, 7, 16, 6, 10, 15, tzinfo=timezone.utc)
    path = make_train_log_path("a", log_dir=tmp_path, when=when)
    assert path == tmp_path / "tier_a_20260716_061015Z.log"


def test_write_log_header_creates_file(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    write_log_header(
        log_path,
        tier="a",
        config_path=Path("configs/llamafactory/tier_a_lora_sft.yaml"),
        command=["llamafactory-cli", "train", "cfg.yaml"],
    )
    text = log_path.read_text(encoding="utf-8")
    assert "tier: a" in text
    assert "llamafactory-cli train cfg.yaml" in text

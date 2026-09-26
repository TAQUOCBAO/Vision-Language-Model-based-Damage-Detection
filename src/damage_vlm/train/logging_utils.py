"""Helpers for persisting training console logs to disk."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from damage_vlm.config import REPO_ROOT

DEFAULT_LOG_DIR = REPO_ROOT / "outputs" / "logs"


def make_train_log_path(
    tier: str,
    *,
    log_dir: Path | None = None,
    when: datetime | None = None,
) -> Path:
    """Return a unique log file path for a training run.

    Example: ``outputs/logs/tier_a_20260716_061015Z.log``
    """
    log_dir = log_dir or DEFAULT_LOG_DIR
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%SZ")
    safe_tier = tier.replace("/", "_")
    return log_dir / f"tier_{safe_tier}_{stamp}.log"


def write_log_header(log_path: Path, *, tier: str, config_path: Path, command: list[str]) -> None:
    """Create the log file and write a short run header."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    lines = [
        f"# damage-vlm training log",
        f"# started_utc: {started}",
        f"# tier: {tier}",
        f"# config: {config_path}",
        f"# command: {' '.join(command)}",
        "#" + "-" * 72,
        "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")

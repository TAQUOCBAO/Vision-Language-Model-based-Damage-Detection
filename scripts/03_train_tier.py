#!/usr/bin/env python3
"""Launch (or dry-run) LLaMA-Factory LoRA training for a tier.

Stdout and stderr are teed to a timestamped log file under outputs/logs/
so training progress survives terminal disconnects.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from damage_vlm.config import REPO_ROOT
from damage_vlm.train.logging_utils import make_train_log_path, write_log_header
from damage_vlm.train.register_dataset import (
    LlamaFactoryNotFoundError,
    build_llamafactory_command,
    load_train_yaml,
    write_dataset_info,
)

# Python 3.14 needs datasets>=4.4 (pickle fix); LLaMA-Factory still pins <=4.0.0.
# expandable_segments reduces fragmentation OOMs during large CE logits allocs.
_TRAIN_ENV_OVERRIDES = {
    "DISABLE_VERSION_CHECK": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

# Peak LoRA step on this 8B VL recipe is ~22GB; leave headroom for CE / fragmentation.
_MIN_FREE_VRAM_GIB = 30.0


def _append_line(log_path: Path, line: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)
        if not line.endswith("\n"):
            f.write("\n")


def check_free_vram_gib(min_free_gib: float = _MIN_FREE_VRAM_GIB) -> float | None:
    """Return free VRAM in GiB, or None if CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free_bytes, _total = torch.cuda.mem_get_info()
    return free_bytes / (1024**3)


def warn_or_fail_low_vram(*, min_free_gib: float, allow_low_vram: bool) -> None:
    free = check_free_vram_gib(min_free_gib)
    if free is None:
        print("WARNING: CUDA not available; cannot check free VRAM.", file=sys.stderr)
        return
    msg = (
        f"Free GPU memory: {free:.1f} GiB "
        f"(recommended >= {min_free_gib:.0f} GiB for Qwen3-VL-8B LoRA).\n"
        "Another process is likely holding VRAM (check: nvidia-smi).\n"
        "On this machine a text-generation-server often uses ~25 GiB — stop it first."
    )
    if free < min_free_gib:
        if allow_low_vram:
            print(f"WARNING: {msg}", file=sys.stderr)
        else:
            print(
                f"ERROR: {msg}\n"
                "Re-run with --allow-low-vram to override (likely to OOM).",
                file=sys.stderr,
            )
            raise SystemExit(2)
    else:
        print(f"Free GPU memory: {free:.1f} GiB")


def run_training_with_log(
    cmd: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run ``cmd``, streaming output to console and appending to ``log_path``."""
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            _append_line(log_path, line.rstrip("\n"))
    finally:
        process.stdout.close()
    return_code = process.wait()
    finished = datetime.now(timezone.utc).isoformat()
    _append_line(log_path, "")
    _append_line(log_path, f"# finished_utc: {finished}")
    _append_line(log_path, f"# exit_code: {return_code}")
    return return_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=["a", "b", "c", "full"],
        required=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional explicit log path (default: outputs/logs/tier_<tier>_<timestamp>Z.log)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "logs",
        help="Directory for auto-named training logs",
    )
    parser.add_argument(
        "--allow-low-vram",
        action="store_true",
        help="Skip the free-VRAM preflight (training may OOM)",
    )
    parser.add_argument(
        "--min-free-vram-gib",
        type=float,
        default=_MIN_FREE_VRAM_GIB,
        help=f"Minimum free GPU GiB before train (default {_MIN_FREE_VRAM_GIB})",
    )
    args = parser.parse_args()

    names = {
        "a": "tier_a_lora_sft.yaml",
        "b": "tier_b_lora_sft.yaml",
        "c": "tier_c_lora_sft.yaml",
        "full": "full_winner_lora_sft.yaml",
    }
    cfg_path = REPO_ROOT / "configs" / "llamafactory" / names[args.tier]
    info_path = write_dataset_info()
    print(f"Registered LLaMA-Factory datasets in {info_path}")

    if not args.dry_run:
        warn_or_fail_low_vram(
            min_free_gib=args.min_free_vram_gib,
            allow_low_vram=args.allow_low_vram,
        )

    try:
        cmd = build_llamafactory_command(cfg_path, dry_run=args.dry_run)
    except LlamaFactoryNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(127) from exc

    print(" ".join(cmd))

    log_path = args.log_file or make_train_log_path(args.tier, log_dir=args.log_dir)
    write_log_header(log_path, tier=args.tier, config_path=cfg_path, command=cmd)
    print(f"Training log: {log_path}")

    if args.dry_run:
        _append_line(log_path, "# dry_run: true (command not executed)")
        return

    # Also record the resolved output_dir from YAML for easier debugging.
    cfg = load_train_yaml(cfg_path)
    output_dir = cfg.get("output_dir")
    if output_dir:
        _append_line(log_path, f"# llamafactory_output_dir: {output_dir}")

    child_env = {**os.environ, **_TRAIN_ENV_OVERRIDES}
    # Preserve user override if they already set alloc conf.
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        child_env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    _append_line(
        log_path,
        "# env: "
        f"DISABLE_VERSION_CHECK={child_env.get('DISABLE_VERSION_CHECK')} "
        f"PYTORCH_CUDA_ALLOC_CONF={child_env.get('PYTORCH_CUDA_ALLOC_CONF')}",
    )

    try:
        code = run_training_with_log(
            cmd,
            log_path=log_path,
            cwd=REPO_ROOT,
            env=child_env,
        )
    except FileNotFoundError as exc:
        hint = (
            f"Failed to start training process: {exc}\n"
            "The LLaMA-Factory CLI is missing from PATH. "
            'Install with: pip install -U "llamafactory[torch,metrics]"'
        )
        print(hint, file=sys.stderr)
        _append_line(log_path, f"# error: {hint}")
        raise SystemExit(127) from exc
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()

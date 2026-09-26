#!/usr/bin/env python3
"""Select winning tier from holdout eval JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from damage_vlm.config import REPO_ROOT
from damage_vlm.select.winner import select_winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evals-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "evals",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "evals" / "winner.json",
    )
    args = parser.parse_args()

    metrics = []
    for path in sorted(args.evals_dir.glob("*.json")):
        if path.name == "winner.json":
            continue
        metrics.append(json.loads(path.read_text(encoding="utf-8")))
    result = select_winner(metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["winner"], indent=2))


if __name__ == "__main__":
    main()

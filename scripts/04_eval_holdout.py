#!/usr/bin/env python3
"""Evaluate a tier on the frozen holdout using silver labels + METEOR.

Without a trained checkpoint, pass --predictions-json produced offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from damage_vlm.config import REPO_ROOT
from damage_vlm.metrics.f1 import multilabel_f1
from damage_vlm.metrics.meteor_score import meteor_mean
from damage_vlm.select.winner import weighted_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument(
        "--silver-json",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "silver_labels.json",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "splits" / "v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    silver = json.loads(args.silver_json.read_text(encoding="utf-8"))
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    val_ids = set(split["val_ids"])
    preds = {p["image_id"]: p for p in json.loads(args.predictions_json.read_text(encoding="utf-8"))}

    y_true = []
    y_pred = []
    refs = []
    hyps = []
    missing = []
    for image_id in sorted(val_ids):
        if image_id not in preds:
            missing.append(image_id)
            continue
        gold = silver[image_id]
        pred = preds[image_id]
        y_true.append(gold["damage_categories"])
        y_pred.append(pred["damage_categories"])
        refs.append(gold["description"])
        hyps.append(pred["description"])

    if missing:
        raise SystemExit(f"Missing predictions for {len(missing)} val ids (e.g. {missing[:3]})")

    f1 = multilabel_f1(y_true, y_pred)
    try:
        meteor = meteor_mean(refs, hyps)
    except Exception as exc:  # noqa: BLE001 — surface METEOR env issues clearly
        print(f"WARNING: METEOR unavailable ({exc}); using 0.0")
        meteor = 0.0

    report = {
        "tier": args.tier,
        "f1": f1,
        "meteor": meteor,
        "weighted_score": weighted_score(f1, meteor),
        "n": len(y_true),
    }
    out = args.out or (REPO_ROOT / "outputs" / "evals" / f"{args.tier}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Winner selection with relative F1 safety gate and weighted score."""

from __future__ import annotations

from typing import Any

from damage_vlm.config import load_eval_config


def weighted_score(f1: float, meteor: float, cfg: dict[str, Any] | None = None) -> float:
    cfg = (cfg or load_eval_config()).get("winner", {})
    return float(cfg.get("f1_weight", 0.6)) * f1 + float(cfg.get("meteor_weight", 0.4)) * meteor


def select_winner(tier_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Select best tier among those within relative 3% of best F1.

    Each item needs keys: ``tier``, ``f1``, ``meteor``.
    """
    if not tier_metrics:
        raise ValueError("No tier metrics provided")

    cfg = load_eval_config()["winner"]
    floor = float(cfg.get("f1_relative_floor", 0.97))
    best_f1 = max(float(m["f1"]) for m in tier_metrics)

    survivors: list[dict[str, Any]] = []
    ranked_all: list[dict[str, Any]] = []
    for m in tier_metrics:
        f1 = float(m["f1"])
        meteor = float(m["meteor"])
        gate_pass = f1 >= best_f1 * floor
        entry = {
            **m,
            "f1": f1,
            "meteor": meteor,
            "weighted_score": weighted_score(f1, meteor),
            "f1_gate_pass": gate_pass,
            "best_f1": best_f1,
        }
        ranked_all.append(entry)
        if gate_pass:
            survivors.append(entry)

    ranked_all.sort(key=lambda x: (-x["weighted_score"], -x["f1"]))
    winner = max(survivors, key=lambda x: (x["weighted_score"], x["f1"]))

    return {
        "winner_tier": winner["tier"],
        "winner": winner,
        "all": ranked_all,
    }

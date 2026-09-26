"""Winner selection tests covering AE3 relative F1 gate."""

from __future__ import annotations

from damage_vlm.select.winner import select_winner, weighted_score


def test_weighted_score_60_40() -> None:
    assert abs(weighted_score(1.0, 0.0) - 0.6) < 1e-9
    assert abs(weighted_score(0.0, 1.0) - 0.4) < 1e-9


def test_ae3_high_meteor_low_f1_discarded() -> None:
    metrics = [
        {"tier": "A", "f1": 0.80, "meteor": 0.40},
        {"tier": "B", "f1": 0.79, "meteor": 0.42},
        {"tier": "C", "f1": 0.70, "meteor": 0.90},  # >3% below best F1
    ]
    result = select_winner(metrics)
    assert result["winner_tier"] in {"A", "B"}
    assert result["winner_tier"] != "C"
    by_tier = {m["tier"]: m for m in result["all"]}
    assert by_tier["C"]["f1_gate_pass"] is False
    assert by_tier["A"]["f1_gate_pass"] is True


def test_empty_f1_zero_division_safe() -> None:
    from damage_vlm.metrics.f1 import multilabel_f1

    score = multilabel_f1([[]], [[]])
    assert score == 0.0

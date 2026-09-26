"""Official Test Requirements metrics: category accuracy + METEOR.

The organizer document scores two axes:
1. Accuracy of damage-category classification vs ground-truth labels.
2. METEOR of generated descriptions vs human reference descriptions.

Because the task is multi-label, this module reports:
- Q1 binary accuracy (any damage vs none)
- subset accuracy (exact category-set match)
- label accuracy (1 − Hamming loss over the 10 closed classes)
- sample-average multi-label F1 (supplementary)
- mean METEOR
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from damage_vlm.metrics.f1 import multilabel_f1, to_indicator_matrix
from damage_vlm.metrics.meteor_score import meteor_mean


def _as_bool_damage(label_lists: Sequence[Iterable[str]], *, explicit: Sequence[bool] | None = None) -> np.ndarray:
    if explicit is not None:
        return np.asarray([bool(x) for x in explicit], dtype=bool)
    return np.asarray([len(list(labels)) > 0 for labels in label_lists], dtype=bool)


def subset_accuracy(
    y_true: Sequence[Iterable[str]],
    y_pred: Sequence[Iterable[str]],
) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true / y_pred length mismatch")
    if not y_true:
        return 0.0
    hits = [set(t) == set(p) for t, p in zip(y_true, y_pred)]
    return float(np.mean(hits))


def label_accuracy(
    y_true: Sequence[Iterable[str]],
    y_pred: Sequence[Iterable[str]],
) -> float:
    """Mean per-label correctness (1 − Hamming loss) over the closed set."""
    yt = to_indicator_matrix(y_true)
    yp = to_indicator_matrix(y_pred)
    return float((yt == yp).mean())


def q1_binary_accuracy(
    y_true: Sequence[Iterable[str]],
    y_pred: Sequence[Iterable[str]],
    *,
    true_has_damage: Sequence[bool] | None = None,
    pred_has_damage: Sequence[bool] | None = None,
) -> float:
    yt = _as_bool_damage(y_true, explicit=true_has_damage)
    yp = _as_bool_damage(y_pred, explicit=pred_has_damage)
    if yt.shape != yp.shape:
        raise ValueError("Q1 gold / pred length mismatch")
    if yt.size == 0:
        return 0.0
    return float((yt == yp).mean())


def official_scores(
    y_true: Sequence[Iterable[str]],
    y_pred: Sequence[Iterable[str]],
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    true_has_damage: Sequence[bool] | None = None,
    pred_has_damage: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Compute the official two-axis scores plus diagnostic multi-label extras."""
    if not (len(y_true) == len(y_pred) == len(references) == len(hypotheses)):
        raise ValueError("All official score sequences must have the same length")

    meteor = meteor_mean(references, hypotheses) if references else 0.0
    return {
        "n": len(y_true),
        "q1_binary_accuracy": q1_binary_accuracy(
            y_true,
            y_pred,
            true_has_damage=true_has_damage,
            pred_has_damage=pred_has_damage,
        ),
        "category_subset_accuracy": subset_accuracy(y_true, y_pred),
        "category_label_accuracy": label_accuracy(y_true, y_pred),
        "category_sample_f1": multilabel_f1(y_true, y_pred),
        "meteor": meteor,
    }

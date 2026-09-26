"""Multi-label F1 over the closed damage category set."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import f1_score

from damage_vlm.config import closed_labels, load_eval_config


def to_indicator_matrix(
    label_lists: Sequence[Iterable[str]],
) -> np.ndarray:
    classes = closed_labels()
    index = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(label_lists), len(classes)), dtype=np.int8)
    for row, labels in enumerate(label_lists):
        for label in labels:
            if label in index:
                y[row, index[label]] = 1
    return y


def multilabel_f1(
    y_true: Sequence[Iterable[str]],
    y_pred: Sequence[Iterable[str]],
) -> float:
    cfg = load_eval_config()["f1"]
    yt = to_indicator_matrix(y_true)
    yp = to_indicator_matrix(y_pred)
    return float(
        f1_score(
            yt,
            yp,
            average=cfg.get("average", "samples"),
            zero_division=cfg.get("zero_division", 0),
        )
    )

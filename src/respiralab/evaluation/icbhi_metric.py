from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

from respiralab.evaluation.metrics import LABEL_ORDER

NORMAL_LABEL = "normal"


def icbhi_score(
    y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...] = LABEL_ORDER
) -> dict:
    if NORMAL_LABEL not in labels:
        raise ValueError(f"labels must include {NORMAL_LABEL!r}, got {labels!r}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    normal_idx = labels.index(NORMAL_LABEL)
    abnormal_idx = [i for i in range(len(labels)) if i != normal_idx]

    n_true_normal = int(cm[normal_idx].sum())
    correct_normal = int(cm[normal_idx, normal_idx])
    specificity = correct_normal / n_true_normal if n_true_normal else 0.0

    n_true_abnormal = int(cm[abnormal_idx, :].sum())
    correct_abnormal = int(sum(cm[i, i] for i in abnormal_idx))
    sensitivity = correct_abnormal / n_true_abnormal if n_true_abnormal else 0.0

    return {
        "specificity": specificity,
        "sensitivity": sensitivity,
        "score": (specificity + sensitivity) / 2,
        "n_true_normal": n_true_normal,
        "n_true_abnormal": n_true_abnormal,
        "correct_normal": correct_normal,
        "correct_abnormal": correct_abnormal,
    }

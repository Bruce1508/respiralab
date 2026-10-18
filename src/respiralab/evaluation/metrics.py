from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

LABEL_ORDER = ("normal", "crackle", "wheeze", "both")


def score_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...] = LABEL_ORDER
) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        },
        "confusion_matrix": {"labels": list(labels), "matrix": cm.tolist()},
    }

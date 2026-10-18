from __future__ import annotations

import pytest
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import precision_recall_fscore_support as sklearn_prfs

from respiralab.evaluation.metrics import LABEL_ORDER, score_predictions

Y_TRUE = ["normal", "normal", "normal", "crackle", "crackle", "wheeze", "both"]
Y_PRED = ["normal", "normal", "crackle", "crackle", "normal", "wheeze", "normal"]


def test_confusion_matrix_matches_hand_count():
    result = score_predictions(Y_TRUE, Y_PRED)
    expected = [
        [2, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ]
    assert result["confusion_matrix"]["matrix"] == expected
    assert result["confusion_matrix"]["labels"] == list(LABEL_ORDER)


def test_confusion_matrix_matches_sklearn_library_output():
    result = score_predictions(Y_TRUE, Y_PRED)
    expected = sklearn_confusion_matrix(Y_TRUE, Y_PRED, labels=LABEL_ORDER).tolist()
    assert result["confusion_matrix"]["matrix"] == expected


def test_per_class_metrics_match_hand_calculation():
    result = score_predictions(Y_TRUE, Y_PRED)

    assert result["per_class"]["normal"]["precision"] == pytest.approx(2 / 4)
    assert result["per_class"]["normal"]["recall"] == pytest.approx(2 / 3)
    assert result["per_class"]["normal"]["support"] == 3

    assert result["per_class"]["crackle"]["precision"] == pytest.approx(1 / 2)
    assert result["per_class"]["crackle"]["recall"] == pytest.approx(1 / 2)
    assert result["per_class"]["crackle"]["support"] == 2

    assert result["per_class"]["wheeze"]["precision"] == pytest.approx(1.0)
    assert result["per_class"]["wheeze"]["recall"] == pytest.approx(1.0)
    assert result["per_class"]["wheeze"]["support"] == 1

    assert result["per_class"]["both"]["precision"] == pytest.approx(0.0)
    assert result["per_class"]["both"]["recall"] == pytest.approx(0.0)
    assert result["per_class"]["both"]["support"] == 1


def test_per_class_metrics_match_sklearn_library_output():
    result = score_predictions(Y_TRUE, Y_PRED)
    precision, recall, f1, support = sklearn_prfs(
        Y_TRUE, Y_PRED, labels=LABEL_ORDER, zero_division=0
    )
    for i, label in enumerate(LABEL_ORDER):
        assert result["per_class"][label]["precision"] == pytest.approx(precision[i])
        assert result["per_class"][label]["recall"] == pytest.approx(recall[i])
        assert result["per_class"][label]["f1"] == pytest.approx(f1[i])
        assert result["per_class"][label]["support"] == int(support[i])


def test_macro_f1_matches_hand_calculation():
    result = score_predictions(Y_TRUE, Y_PRED)
    per_class_f1 = [result["per_class"][label]["f1"] for label in LABEL_ORDER]
    assert result["macro_f1"] == pytest.approx(sum(per_class_f1) / len(LABEL_ORDER))


def test_macro_f1_matches_sklearn_library_output():
    result = score_predictions(Y_TRUE, Y_PRED)
    expected = sklearn_f1_score(
        Y_TRUE, Y_PRED, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    assert result["macro_f1"] == pytest.approx(expected)

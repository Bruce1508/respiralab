from __future__ import annotations

import pytest

from respiralab.evaluation.icbhi_metric import icbhi_score

Y_TRUE = (
    ["normal"] * 5
    + ["crackle"] * 6
    + ["wheeze"] * 2
    + ["both"] * 2
)
Y_PRED = (
    ["normal", "normal", "normal", "normal", "crackle"]
    + ["crackle"] * 6
    + ["crackle", "crackle"]
    + ["both", "normal"]
)


def test_specificity_matches_hand_calculation():
    result = icbhi_score(Y_TRUE, Y_PRED)
    assert result["n_true_normal"] == 5
    assert result["correct_normal"] == 4
    assert result["specificity"] == pytest.approx(4 / 5)


def test_pooled_sensitivity_matches_hand_calculation():
    result = icbhi_score(Y_TRUE, Y_PRED)
    assert result["n_true_abnormal"] == 10
    assert result["correct_abnormal"] == 7
    assert result["sensitivity"] == pytest.approx(7 / 10)


def test_score_is_mean_of_specificity_and_sensitivity():
    result = icbhi_score(Y_TRUE, Y_PRED)
    assert result["score"] == pytest.approx((4 / 5 + 7 / 10) / 2)


def test_pooled_sensitivity_is_not_macro_recall_across_abnormal_classes():
    result = icbhi_score(Y_TRUE, Y_PRED)
    macro_recall_abnormal = (6 / 6 + 0 / 2 + 1 / 2) / 3
    assert result["sensitivity"] != pytest.approx(macro_recall_abnormal)
    assert macro_recall_abnormal == pytest.approx(0.5)
    assert result["sensitivity"] == pytest.approx(0.7)


def test_result_independent_of_label_tuple_order():
    reordered = icbhi_score(Y_TRUE, Y_PRED, labels=("both", "wheeze", "crackle", "normal"))
    default = icbhi_score(Y_TRUE, Y_PRED)
    assert reordered["specificity"] == pytest.approx(default["specificity"])
    assert reordered["sensitivity"] == pytest.approx(default["sensitivity"])
    assert reordered["score"] == pytest.approx(default["score"])


def test_raises_when_normal_label_missing():
    with pytest.raises(ValueError, match="normal"):
        icbhi_score(Y_TRUE, Y_PRED, labels=("crackle", "wheeze", "both"))

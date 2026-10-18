from __future__ import annotations

import numpy as np
import pytest

from respiralab.evaluation.bootstrap import patient_bootstrap_ci


def _accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def test_point_estimate_matches_metric_on_full_data():
    patient_ids = ["p1", "p1", "p2", "p2", "p3"]
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 0, 0]
    result = patient_bootstrap_ci(patient_ids, y_true, y_pred, _accuracy, n_replicates=200, seed=1)
    assert result["point_estimate"] == pytest.approx(_accuracy(y_true, y_pred))
    assert result["n_patients"] == 3


def test_deterministic_given_same_seed():
    patient_ids = ["p1"] * 10 + ["p2"] * 10 + ["p3"] * 10
    y_true = [1] * 30
    y_pred = [1] * 15 + [0] * 15
    r1 = patient_bootstrap_ci(patient_ids, y_true, y_pred, _accuracy, n_replicates=300, seed=7)
    r2 = patient_bootstrap_ci(patient_ids, y_true, y_pred, _accuracy, n_replicates=300, seed=7)
    assert r1["ci_lower"] == pytest.approx(r2["ci_lower"])
    assert r1["ci_upper"] == pytest.approx(r2["ci_upper"])


def test_bootstrap_groups_by_patient_not_by_cycle():
    patient_ids = ["A"] * 50 + ["B"] * 50
    y_true = [1] * 100
    y_pred = [1] * 50 + [0] * 50

    result = patient_bootstrap_ci(
        patient_ids, y_true, y_pred, _accuracy, n_replicates=2000, seed=42
    )

    assert result["n_patients"] == 2
    assert result["point_estimate"] == pytest.approx(0.5)
    assert result["ci_lower"] < 0.1
    assert result["ci_upper"] > 0.9

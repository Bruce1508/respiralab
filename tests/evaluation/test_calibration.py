from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from respiralab.evaluation.calibration import (
    collect_oof_logits,
    expected_calibration_error,
    fit_temperature,
    softmax,
)
from respiralab.evaluation.group_cv import make_folds
from respiralab.evaluation.metrics import LABEL_ORDER


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 0.5, -1.0], [0.0, 0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert (probs >= 0).all()


def test_fit_temperature_recovers_known_temperature_on_synthetic_logits():
    rng = np.random.default_rng(0)
    n = 4000
    base_logits = rng.normal(scale=1.5, size=(n, len(LABEL_ORDER)))
    base_probs = softmax(base_logits)
    y_idx = np.array([rng.choice(len(LABEL_ORDER), p=p) for p in base_probs])
    y_true = np.array(LABEL_ORDER)[y_idx]

    true_temperature = 2.5
    # Labels were sampled to be well-calibrated w.r.t. base_logits; scaling by
    # true_temperature makes the "observed" logits overconfident by that factor.
    observed_logits = base_logits * true_temperature

    fitted = fit_temperature(observed_logits, y_true)

    assert fitted == pytest.approx(true_temperature, rel=0.15)


def test_fit_temperature_is_noop_when_already_calibrated():
    rng = np.random.default_rng(1)
    n = 3000
    base_logits = rng.normal(scale=1.0, size=(n, len(LABEL_ORDER)))
    base_probs = softmax(base_logits)
    y_idx = np.array([rng.choice(len(LABEL_ORDER), p=p) for p in base_probs])
    y_true = np.array(LABEL_ORDER)[y_idx]

    fitted = fit_temperature(base_logits, y_true)

    assert fitted == pytest.approx(1.0, abs=0.2)


def test_expected_calibration_error_near_zero_for_perfect_calibration():
    rng = np.random.default_rng(2)
    n = 5000
    logits = rng.normal(scale=1.0, size=(n, len(LABEL_ORDER)))
    probs = softmax(logits)
    y_idx = np.array([rng.choice(len(LABEL_ORDER), p=p) for p in probs])
    y_true = np.array(LABEL_ORDER)[y_idx]

    result = expected_calibration_error(probs, y_true, n_bins=10)

    assert result["ece"] < 0.05
    assert len(result["bins"]) == 10


def test_expected_calibration_error_decreases_after_temperature_scaling():
    rng = np.random.default_rng(3)
    n = 4000
    base_logits = rng.normal(scale=1.5, size=(n, len(LABEL_ORDER)))
    base_probs = softmax(base_logits)
    y_idx = np.array([rng.choice(len(LABEL_ORDER), p=p) for p in base_probs])
    y_true = np.array(LABEL_ORDER)[y_idx]
    overconfident_logits = base_logits * 3.0

    ece_before = expected_calibration_error(softmax(overconfident_logits), y_true)["ece"]
    temperature = fit_temperature(overconfident_logits, y_true)
    ece_after = expected_calibration_error(softmax(overconfident_logits / temperature), y_true)["ece"]

    assert ece_after < ece_before


def test_collect_oof_logits_reorders_to_label_order_and_fills_every_sample():
    rng = np.random.default_rng(4)
    n_patients, cycles_per_patient = 20, 6
    patient_ids = np.repeat([f"p{i}" for i in range(n_patients)], cycles_per_patient)
    n = len(patient_ids)
    X = rng.normal(size=(n, 3))
    y = rng.choice(LABEL_ORDER, size=n)
    folds = make_folds(patient_ids, y, n_splits=4, seed=42)

    def build_fn():
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])

    logits, y_out = collect_oof_logits(X, y, build_fn, folds)

    assert logits.shape == (n, len(LABEL_ORDER))
    assert not np.isnan(logits).any()
    assert (y_out == y).all()


def test_collect_oof_logits_raises_if_a_sample_is_never_held_out():
    rng = np.random.default_rng(5)
    n = 20
    X = rng.normal(size=(n, 3))
    y = rng.choice(LABEL_ORDER, size=n)
    # deliberately incomplete folds: last sample never appears in any val_idx
    folds = [(np.arange(10), np.arange(10, 19))]

    def build_fn():
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])

    with pytest.raises(AssertionError):
        collect_oof_logits(X, y, build_fn, folds)

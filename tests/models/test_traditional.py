from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from respiralab.models.traditional import build_pipeline, evaluate_pipeline


def _linearly_separable_data(n_per_class: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = ("normal", "crackle", "wheeze", "both")
    X, y = ([], [])
    for i, label in enumerate(labels):
        X.append(rng.normal(loc=i * 10.0, scale=0.5, size=(n_per_class, 3)))
        y.extend([label] * n_per_class)
    return (np.vstack(X), np.array(y))


def test_build_pipeline_is_scaler_then_logreg():
    pipeline = build_pipeline(seed=42)
    assert list(pipeline.named_steps) == ["scaler", "clf"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["clf"], LogisticRegression)


def test_scaler_fits_only_on_train_not_dev():
    X_train, y_train = _linearly_separable_data(n_per_class=8, seed=1)
    X_dev, y_dev = _linearly_separable_data(n_per_class=4, seed=2)
    X_dev = X_dev + 1000.0
    pipeline = build_pipeline(seed=42)
    evaluate_pipeline(pipeline, X_train, y_train, X_dev, y_dev)
    fitted_mean = pipeline.named_steps["scaler"].mean_
    assert np.allclose(fitted_mean, X_train.mean(axis=0))
    combined_mean = np.vstack([X_train, X_dev]).mean(axis=0)
    assert not np.allclose(fitted_mean, combined_mean)


def test_evaluate_pipeline_returns_dev_scores_shape():
    X_train, y_train = _linearly_separable_data(n_per_class=8, seed=1)
    X_dev, y_dev = _linearly_separable_data(n_per_class=4, seed=2)
    result = evaluate_pipeline(build_pipeline(seed=42), X_train, y_train, X_dev, y_dev)
    assert result["strategy"] == "logistic_regression"
    assert set(result["per_class"]) == {"normal", "crackle", "wheeze", "both"}
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert result["macro_f1"] > 0.5


def test_reproducible_given_same_seed():
    X_train, y_train = _linearly_separable_data(n_per_class=8, seed=1)
    X_dev, y_dev = _linearly_separable_data(n_per_class=4, seed=2)
    r1 = evaluate_pipeline(build_pipeline(seed=42), X_train, y_train, X_dev, y_dev)
    r2 = evaluate_pipeline(build_pipeline(seed=42), X_train, y_train, X_dev, y_dev)
    assert r1["confusion_matrix"]["matrix"] == r2["confusion_matrix"]["matrix"]
    assert r1["macro_f1"] == pytest.approx(r2["macro_f1"])

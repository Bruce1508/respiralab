"""D17 tests: bounded RBF SVM hyperparameter search (E02), selected on val only.

Synthetic data only — real-data run is a separate CLI evidence step per
the TDD convention used D08-D16.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from respiralab.models.svm_search import build_svm_pipeline, load_search_config, run_search


def _linearly_separable_data(n_per_class: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = ("normal", "crackle", "wheeze", "both")
    X, y = [], []
    for i, label in enumerate(labels):
        X.append(rng.normal(loc=i * 10.0, scale=0.5, size=(n_per_class, 3)))
        y.extend([label] * n_per_class)
    return np.vstack(X), np.array(y)


def test_load_search_config_parses_yaml(tmp_path):
    cfg_path = tmp_path / "search_config.yaml"
    cfg_path.write_text("model: svm_rbf\nseed: 42\nparam_grid:\n  C: [0.1, 1.0]\n  gamma: [scale, 0.1]\n")

    config = load_search_config(cfg_path)

    assert config["seed"] == 42
    assert config["param_grid"]["C"] == [0.1, 1.0]
    assert config["param_grid"]["gamma"] == ["scale", 0.1]


def test_build_svm_pipeline_is_scaler_then_rbf_svc():
    pipeline = build_svm_pipeline(C=1.0, gamma="scale", seed=42)
    assert list(pipeline.named_steps) == ["scaler", "clf"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    clf = pipeline.named_steps["clf"]
    assert isinstance(clf, SVC)
    assert clf.kernel == "rbf"
    assert clf.C == 1.0
    assert clf.gamma == "scale"


def test_run_search_tries_every_grid_combination():
    X_train, y_train = _linearly_separable_data(n_per_class=10, seed=1)
    X_dev, y_dev = _linearly_separable_data(n_per_class=5, seed=2)
    config = {"seed": 42, "param_grid": {"C": [0.1, 1.0], "gamma": ["scale", 0.1]}}

    result = run_search(config, X_train, y_train, X_dev, y_dev)

    assert result["n_trials"] == 4  # 2 C values x 2 gamma values, full grid
    assert len(result["trials"]) == 4
    assert {(t["C"], t["gamma"]) for t in result["trials"]} == {
        (0.1, "scale"),
        (0.1, 0.1),
        (1.0, "scale"),
        (1.0, 0.1),
    }


def test_run_search_selects_best_by_val_macro_f1_only():
    X_train, y_train = _linearly_separable_data(n_per_class=10, seed=1)
    X_dev, y_dev = _linearly_separable_data(n_per_class=5, seed=2)
    config = {"seed": 42, "param_grid": {"C": [0.001, 1.0, 100.0], "gamma": ["scale"]}}

    result = run_search(config, X_train, y_train, X_dev, y_dev)

    best_macro_f1 = max(t["macro_f1"] for t in result["trials"])
    assert result["best"]["macro_f1"] == pytest.approx(best_macro_f1)
    assert result["best"]["strategy"] == "svm_rbf"
    # with cleanly separated classes a well-fit SVM should score well above chance
    assert result["best"]["macro_f1"] > 0.5

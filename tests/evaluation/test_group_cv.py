from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from respiralab.evaluation.group_cv import (
    assert_no_patient_crosses_folds,
    load_trainval_pool,
    make_folds,
    run_grouped_cv,
)

LABELS = ("normal", "crackle", "wheeze", "both")


def _synthetic_pool(n_patients: int = 40, cycles_per_patient: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    patient_ids = np.repeat([f"p{i}" for i in range(n_patients)], cycles_per_patient)
    n = len(patient_ids)
    X = rng.normal(size=(n, 3))
    y = rng.choice(LABELS, size=n)
    return X, y, patient_ids


def test_make_folds_keeps_every_patient_in_exactly_one_fold():
    _, y, patient_ids = _synthetic_pool()
    folds = make_folds(patient_ids, y, n_splits=5, seed=42)
    assert len(folds) == 5
    assert_no_patient_crosses_folds(patient_ids, folds)
    # every val fold union covers all patients exactly once
    all_val_patients = np.concatenate([patient_ids[val_idx] for _, val_idx in folds])
    assert sorted(set(all_val_patients)) == sorted(set(patient_ids))
    assert len(all_val_patients) == len(patient_ids)


def test_assert_no_patient_crosses_folds_raises_on_violation():
    patient_ids = np.array(["p1", "p1", "p2", "p2"])
    bad_folds = [
        (np.array([0]), np.array([1, 2])),  # "p1" appears in fold 0's val (idx 1) ...
        (np.array([0]), np.array([0, 3])),  # ... and fold 1's val (idx 0)
    ]
    with pytest.raises(AssertionError):
        assert_no_patient_crosses_folds(patient_ids, bad_folds)


def test_run_grouped_cv_reports_mean_and_sd_across_folds():
    X, y, patient_ids = _synthetic_pool(n_patients=40, cycles_per_patient=5, seed=1)
    folds = make_folds(patient_ids, y, n_splits=5, seed=42)

    def build_fn():
        return DummyClassifier(strategy="most_frequent")

    result = run_grouped_cv(X, y, patient_ids, build_fn, folds)
    assert result["n_splits"] == 5
    assert len(result["folds"]) == 5
    assert 0.0 <= result["macro_f1_mean"] <= 1.0
    assert result["macro_f1_sd"] >= 0.0
    assert set(result["per_class_recall_mean"]) == set(LABELS)
    for fold in result["folds"]:
        assert fold["n_val_cycles"] > 0
        assert fold["n_train_patients"] + fold["n_val_patients"] == 40


def test_load_trainval_pool_excludes_test_split(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    rows = []
    for split, patient, label in [
        ("train", "p1", "normal"),
        ("val", "p2", "crackle"),
        ("test", "p3", "wheeze"),
    ]:
        fname = f"{patient}.npy"
        np.save(cache_dir / fname, np.array([1.0, 2.0]))
        rows.append(
            {"patient_id": patient, "split": split, "label4": label, "cache_path": fname}
        )
    manifest_df = pd.DataFrame(rows)

    X, y, patient_ids = load_trainval_pool(manifest_df, cache_dir)

    assert set(patient_ids) == {"p1", "p2"}
    assert "p3" not in patient_ids
    assert len(X) == len(y) == len(patient_ids) == 2


def test_load_trainval_pool_raises_on_empty(tmp_path):
    manifest_df = pd.DataFrame(
        [{"patient_id": "p1", "split": "test", "label4": "normal", "cache_path": "x.npy"}]
    )
    with pytest.raises(ValueError):
        load_trainval_pool(manifest_df, tmp_path)

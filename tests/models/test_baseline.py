from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from respiralab.models.baseline import evaluate_dummy, load_split_features, patient_counts


def _write_cache(cache_dir, cycle_id: str, vec: np.ndarray) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rel = f"{cycle_id}.npy"
    np.save(cache_dir / rel, vec)
    return rel


def _manifest(cache_dir) -> pd.DataFrame:
    rows = []
    for i in range(6):
        cid = f"train_normal_{i}"
        rel = _write_cache(cache_dir, cid, np.full(4, float(i)))
        rows.append(
            {
                "cycle_id": cid,
                "patient_id": f"p{i % 3}",
                "split": "train",
                "label4": "normal",
                "cache_path": rel,
            }
        )
    for i in range(2):
        cid = f"train_crackle_{i}"
        rel = _write_cache(cache_dir, cid, np.full(4, float(i)))
        rows.append(
            {
                "cycle_id": cid,
                "patient_id": "p9",
                "split": "train",
                "label4": "crackle",
                "cache_path": rel,
            }
        )
    for i, label in enumerate(["normal", "normal", "wheeze"]):
        cid = f"val_{label}_{i}"
        rel = _write_cache(cache_dir, cid, np.full(4, float(i)))
        rows.append(
            {
                "cycle_id": cid,
                "patient_id": f"v{i}",
                "split": "val",
                "label4": label,
                "cache_path": rel,
            }
        )
    return pd.DataFrame(rows)


def test_load_split_features_returns_matching_shapes_and_labels(tmp_path):
    manifest_df = _manifest(tmp_path)
    X, y = load_split_features(manifest_df, "train", tmp_path)
    assert X.shape == (8, 4)
    assert list(y) == ["normal"] * 6 + ["crackle"] * 2


def test_load_split_features_raises_on_missing_split(tmp_path):
    manifest_df = _manifest(tmp_path)
    with pytest.raises(ValueError, match="test"):
        load_split_features(manifest_df, "test", tmp_path)


def test_majority_baseline_always_predicts_train_majority_label(tmp_path):
    manifest_df = _manifest(tmp_path)
    X_train, y_train = load_split_features(manifest_df, "train", tmp_path)
    X_val, y_val = load_split_features(manifest_df, "val", tmp_path)
    result = evaluate_dummy("majority", "most_frequent", X_train, y_train, X_val, y_val)
    assert result["strategy"] == "majority"
    assert result["per_class"]["normal"]["recall"] == pytest.approx(1.0)
    assert result["per_class"]["wheeze"]["recall"] == pytest.approx(0.0)
    assert 0.0 <= result["macro_f1"] <= 1.0


def test_stratified_baseline_is_deterministic_given_same_seed(tmp_path):
    manifest_df = _manifest(tmp_path)
    X_train, y_train = load_split_features(manifest_df, "train", tmp_path)
    X_val, y_val = load_split_features(manifest_df, "val", tmp_path)
    r1 = evaluate_dummy("stratified", "stratified", X_train, y_train, X_val, y_val)
    r2 = evaluate_dummy("stratified", "stratified", X_train, y_train, X_val, y_val)
    assert r1["confusion_matrix"]["matrix"] == r2["confusion_matrix"]["matrix"]


def test_patient_counts_counts_unique_patients_per_split(tmp_path):
    manifest_df = _manifest(tmp_path)
    counts = patient_counts(manifest_df)
    assert counts["train"] == 4
    assert counts["val"] == 3
    assert counts["test"] == 0

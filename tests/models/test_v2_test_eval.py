from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from respiralab.models import v2_test_eval


def test_build_test_rows_joins_cycle_bounds_for_test_split_only():
    manifest_df = pd.DataFrame(
        {
            "cycle_id": ["c1", "c2", "c3"],
            "patient_id": ["p1", "p2", "p3"],
            "label4": ["normal", "crackle", "wheeze"],
            "split": ["test", "train", "test"],
        }
    )
    cycles_df = pd.DataFrame(
        {
            "cycle_id": ["c1", "c2", "c3"],
            "record_id": ["r1", "r2", "r3"],
            "start_s": [0.0, 0.0, 1.0],
            "end_s": [2.0, 2.0, 3.0],
        }
    )

    rows = v2_test_eval.build_test_rows(manifest_df, cycles_df)

    assert sorted(rows["cycle_id"]) == ["c1", "c3"]
    assert set(rows["split"]) == {"test"}
    assert "start_s" in rows.columns and "record_id" in rows.columns


def test_build_test_rows_raises_on_missing_cycle_bounds():
    manifest_df = pd.DataFrame({"cycle_id": ["missing"], "patient_id": ["p1"], "label4": ["normal"], "split": ["test"]})
    cycles_df = pd.DataFrame({"cycle_id": [], "record_id": [], "start_s": [], "end_s": []})

    with pytest.raises(ValueError, match="missing from cycles.csv"):
        v2_test_eval.build_test_rows(manifest_df, cycles_df)


def test_build_test_rows_raises_when_no_test_rows():
    manifest_df = pd.DataFrame({"cycle_id": ["c1"], "patient_id": ["p1"], "label4": ["normal"], "split": ["train"]})
    cycles_df = pd.DataFrame({"cycle_id": ["c1"], "record_id": ["r1"], "start_s": [0.0], "end_s": [1.0]})

    with pytest.raises(ValueError, match="split='test'"):
        v2_test_eval.build_test_rows(manifest_df, cycles_df)


def _write_cache(cache_dir: Path, name: str, vec: np.ndarray) -> str:
    path = cache_dir / f"{name}.npy"
    np.save(path, vec)
    return f"{name}.npy"


def test_evaluate_e01_test_fits_train_only_and_applies_temperature(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    cache_dir = tmp_path
    labels = ["normal", "crackle", "wheeze", "both"]
    for split, n_per_class in (("train", 10), ("test", 4)):
        for label_idx, label in enumerate(labels):
            for i in range(n_per_class):
                vec = rng.normal(loc=label_idx * 2.0, scale=0.3, size=(8,)).astype(np.float32)
                name = f"{split}_{label}_{i}"
                cache_path = _write_cache(cache_dir, name, vec)
                rows.append(
                    {
                        "cache_path": cache_path,
                        "label4": label,
                        "split": split,
                        "patient_id": f"{split}_{label}_{i}",
                    }
                )
    manifest_df = pd.DataFrame(rows)

    result = v2_test_eval.evaluate_e01_test(manifest_df, cache_dir, calibration_temperature=2.0)

    assert result["model"].startswith("logistic_regression")
    assert result["n_test_cycles"] == 16
    assert 0.0 <= result["macro_f1_test"] <= 1.0
    assert set(result["per_class_test"].keys()) == set(labels)
    assert result["bootstrap_macro_f1"]["ci_lower"] <= result["bootstrap_macro_f1"]["point_estimate"]
    assert result["bootstrap_macro_f1"]["point_estimate"] <= result["bootstrap_macro_f1"]["ci_upper"]
    assert result["calibration_temperature"] == 2.0
    assert result["ece_before"] >= 0.0
    assert result["ece_after"] >= 0.0


def test_evaluate_v2_test_computes_metrics_from_collected_logits(monkeypatch):
    rows_test = pd.DataFrame(
        {
            "cycle_id": ["c1", "c2", "c3", "c4"],
            "patient_id": ["p1", "p1", "p2", "p2"],
            "label4": ["normal", "crackle", "wheeze", "both"],
        }
    )
    fake_logits = np.array(
        [
            [5.0, 0.0, 0.0, 0.0],  # -> normal, correct
            [0.0, 5.0, 0.0, 0.0],  # -> crackle, correct
            [5.0, 0.0, 0.0, 0.0],  # -> normal, wrong (true wheeze)
            [0.0, 0.0, 0.0, 5.0],  # -> both, correct
        ]
    )
    fake_labels = np.array(["normal", "crackle", "wheeze", "both"])

    class _FakeModel:
        def load_state_dict(self, _state, strict=True):
            pass

        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(v2_test_eval, "ASTClassifier", lambda unfreeze_last_n: _FakeModel())
    monkeypatch.setattr(v2_test_eval.torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(v2_test_eval, "collect_logits", lambda model, loader, device: (fake_logits, fake_labels))

    result = v2_test_eval.evaluate_v2_test(
        rows_test,
        audio_dir=Path("unused"),
        weights_path=Path("unused.pt"),
        calibration_temperature=1.5,
        device="cpu",
        feature_extractor=None,
    )

    assert result["n_test_cycles"] == 4
    # normal: 1 TP + 1 FP (misclassified wheeze) -> precision .5, recall 1, f1=2/3
    # crackle: perfect f1=1; wheeze: 0 predicted -> f1=0; both: perfect f1=1
    assert result["macro_f1_test"] == pytest.approx((2 / 3 + 1 + 0 + 1) / 4, abs=1e-6)
    assert result["per_class_test"]["wheeze"]["recall"] == 0.0
    assert result["bootstrap_macro_f1"]["n_patients"] == 2
    assert result["calibration_temperature"] == 1.5

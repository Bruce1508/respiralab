from __future__ import annotations

import numpy as np
import pandas as pd

from respiralab.models import v2_freeze


def test_calibrate_aggregates_logits_and_labels_across_all_folds(monkeypatch):
    fake_runs = {
        0: {"best_val_macro_f1": 0.4, "val_logits": np.array([[1.0, 0.0], [0.0, 1.0]]), "val_labels": np.array(["normal", "crackle"])},
        1: {"best_val_macro_f1": 0.5, "val_logits": np.array([[0.5, 0.5]]), "val_labels": np.array(["wheeze"])},
    }

    def fake_run_fold(rows, fold, model_seed, args, device, feature_extractor, track_best_state):
        assert track_best_state is True
        assert model_seed == v2_freeze.FINAL_MODEL_SEED
        return fake_runs[fold]

    monkeypatch.setattr(v2_freeze, "run_fold", fake_run_fold)
    monkeypatch.setattr(v2_freeze, "N_SPLITS", 2)
    monkeypatch.setattr(
        v2_freeze,
        "fit_temperature",
        lambda logits, y_true: 1.5,
    )
    monkeypatch.setattr(
        v2_freeze,
        "expected_calibration_error",
        lambda probs, y_true: {"ece": 0.1},
    )

    result = v2_freeze.calibrate(pd.DataFrame(), audio_dir=None, device=None, feature_extractor=None)

    assert result["n_oof_cycles"] == 3
    assert result["temperature"] == 1.5
    assert result["ece_before"] == 0.1
    assert result["ece_after"] == 0.1

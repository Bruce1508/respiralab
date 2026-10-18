from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from respiralab.evaluation.metrics import LABEL_ORDER
from respiralab.inference.predict import (
    InputError,
    analyze_cycle,
    analyze_recording,
    validate_wav,
)


def _toy_pipeline():
    # same class order sklearn actually produces: alphabetical, NOT LABEL_ORDER
    rng = np.random.RandomState(0)
    X = rng.randn(40, 46)
    y = np.array(["normal", "crackle", "wheeze", "both"] * 10)
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced"))]
    )
    pipeline.fit(X, y)
    return pipeline


def _tone(duration_s=1.0, sample_rate=4000, freq=300.0):
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    return 0.1 * np.sin(2 * np.pi * freq * t), sample_rate


def test_validate_wav_accepts_normal_audio():
    audio, sr = _tone()
    validate_wav(audio, sr)  # must not raise


@pytest.mark.parametrize(
    "audio,sample_rate,match",
    [
        (np.array([]), 4000, "empty"),
        (np.zeros(4000), 4000, "silent"),
        (np.array([np.nan] * 4000), 4000, "non-finite"),
        (np.ones((4000, 2)), 4000, "mono"),
        (np.ones(4000), 0, "sample rate"),
        (np.ones(4000 * 61), 4000, "too long"),
    ],
)
def test_validate_wav_rejects_malformed_input(audio, sample_rate, match):
    with pytest.raises(InputError, match=match):
        validate_wav(audio, sample_rate)


def test_analyze_cycle_probabilities_align_with_label_names_not_sklearn_class_order():
    pipeline = _toy_pipeline()
    assert list(pipeline.named_steps["clf"].classes_) != list(LABEL_ORDER), (
        "toy pipeline must reproduce sklearn's real alphabetical class order for this test to be meaningful"
    )
    audio, sr = _tone()
    result = analyze_cycle(pipeline, audio, sr)

    assert result["label_names"] == list(LABEL_ORDER)
    assert len(result["probabilities"]) == 4
    assert pytest.approx(sum(result["probabilities"]), abs=1e-6) == 1.0

    # the label with the highest returned probability must match predict()'s own answer
    predicted_label = result["label_names"][int(np.argmax(result["probabilities"]))]
    direct_prediction = pipeline.predict(
        np.asarray(result["_debug_features"]).reshape(1, -1)
    )[0]
    assert predicted_label == direct_prediction


def test_analyze_cycle_sets_annotated_cycle_mode_and_no_diagnosis_warning():
    pipeline = _toy_pipeline()
    audio, sr = _tone()
    result = analyze_cycle(pipeline, audio, sr)
    assert result["analysis_mode"] == "annotated_cycle"
    assert "not a diagnostic" in result["warning"].lower()


def test_analyze_recording_without_bounds_is_exploratory_with_extra_warning():
    pipeline = _toy_pipeline()
    audio, sr = _tone(duration_s=3.0)
    result = analyze_recording(pipeline, audio, sr, bounds=None)
    assert result["analysis_mode"] == "exploratory_window"
    assert "exploratory" in result["warning"].lower()


def test_analyze_recording_with_valid_bounds_is_annotated_cycle():
    pipeline = _toy_pipeline()
    audio, sr = _tone(duration_s=3.0)
    result = analyze_recording(pipeline, audio, sr, bounds=(0.5, 1.5))
    assert result["analysis_mode"] == "annotated_cycle"


def test_analyze_recording_rejects_invalid_bounds():
    pipeline = _toy_pipeline()
    audio, sr = _tone(duration_s=1.0)
    with pytest.raises(InputError, match="bounds"):
        analyze_recording(pipeline, audio, sr, bounds=(0.9, 0.1))


@pytest.mark.skipif(
    not Path("artifacts/features_cache").is_dir(),
    reason="requires the real ICBHI feature cache (gitignored, not present on a fresh clone/CI)",
)
def test_build_frozen_predictor_reproduces_committed_val_macro_f1():
    from respiralab.evaluation.metrics import score_predictions
    from respiralab.inference.predict import build_frozen_predictor
    from respiralab.models.baseline import load_split_features

    manifest_df = pd.read_csv("manifests/feature_manifest.csv")
    pipeline = build_frozen_predictor("manifests/feature_manifest.csv", "artifacts/features_cache")
    X_val, y_val = load_split_features(manifest_df, "val", "artifacts/features_cache")
    y_pred = pipeline.predict(X_val)
    scores = score_predictions(y_val, y_pred)
    assert scores["macro_f1"] == pytest.approx(0.3782150198602205, abs=1e-6)

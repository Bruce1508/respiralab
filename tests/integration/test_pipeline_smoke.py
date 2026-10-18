from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from respiralab.dsp.features import extract_features
from respiralab.dsp.resample import resample_audio
from respiralab.dsp.segment import extract_cycle
from respiralab.evaluation.metrics import LABEL_ORDER, score_predictions
from respiralab.inference.predict import InputError, analyze_cycle, validate_wav
from respiralab.models.traditional import build_pipeline

TARGET_SR = 4000


def _tone_wav(path, duration_s: float, freq: float, sample_rate: int = 44100) -> None:
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), tone, sample_rate)


# a different tone frequency per class gives the tiny synthetic pipeline a
# trivially learnable signal, so this is a real smoke test, not a no-op
LABEL_FREQ = {"normal": 150.0, "crackle": 600.0, "wheeze": 1200.0, "both": 2000.0}


def _build_synthetic_cycles(tmp_path, n_per_class: int = 6) -> pd.DataFrame:
    rows = []
    for label, freq in LABEL_FREQ.items():
        for i in range(n_per_class):
            record_id = f"{label}_{i}"
            _tone_wav(tmp_path / f"{record_id}.wav", duration_s=1.0, freq=freq)
            rows.append(
                {
                    "cycle_id": f"{record_id}_c0",
                    "record_id": record_id,
                    "patient_id": f"p_{label}_{i}",
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "label4": label,
                }
            )
    return pd.DataFrame(rows)


def _extract_feature_matrix(cycles_df: pd.DataFrame, audio_dir) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _, row in cycles_df.iterrows():
        audio, native_sr = sf.read(str(audio_dir / f"{row['record_id']}.wav"))
        cycle_audio = extract_cycle(audio, native_sr, row["start_s"], row["end_s"])
        resampled = resample_audio(cycle_audio, native_sr, TARGET_SR)
        X.append(extract_features(resampled, TARGET_SR))
        y.append(row["label4"])
    return np.stack(X), np.array(y)


@pytest.fixture
def synthetic_cycles(tmp_path):
    return _build_synthetic_cycles(tmp_path), tmp_path


def test_full_pipeline_runs_end_to_end_on_synthetic_data_only(synthetic_cycles):
    """ingest -> split -> features -> train -> predict, no ICBHI data required."""
    cycles_df, audio_dir = synthetic_cycles

    # patient-level split, same discipline as the real D07 split (no leakage)
    unique_patients = sorted(cycles_df["patient_id"].unique())
    rng = np.random.RandomState(0)
    rng.shuffle(unique_patients)
    split_point = int(len(unique_patients) * 0.7)
    train_patients, val_patients = set(unique_patients[:split_point]), set(unique_patients[split_point:])
    assert not (train_patients & val_patients)

    train_df = cycles_df[cycles_df["patient_id"].isin(train_patients)]
    val_df = cycles_df[cycles_df["patient_id"].isin(val_patients)]
    assert len(train_df) > 0 and len(val_df) > 0

    X_train, y_train = _extract_feature_matrix(train_df, audio_dir)
    X_val, y_val = _extract_feature_matrix(val_df, audio_dir)

    pipeline = build_pipeline(seed=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    scores = score_predictions(y_val, y_pred)

    # trivially separable synthetic tones -> the pipeline must actually learn something
    assert scores["macro_f1"] > 0.5

    # full offline-inference path (D35) on a fresh synthetic WAV, same frozen pipeline type
    audio, native_sr = sf.read(str(audio_dir / "normal_0.wav"))
    resampled = resample_audio(audio, native_sr, TARGET_SR)
    result = analyze_cycle(pipeline, resampled, TARGET_SR)
    assert result["label_names"] == list(LABEL_ORDER)
    assert pytest.approx(sum(result["probabilities"]), abs=1e-6) == 1.0
    assert "not a diagnostic" in result["warning"].lower()


def test_pipeline_rejects_corrupt_input_without_crashing():
    with pytest.raises(InputError):
        validate_wav(np.array([]), 4000)

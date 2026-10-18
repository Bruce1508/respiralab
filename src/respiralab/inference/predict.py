from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from respiralab.dsp.features import extract_features
from respiralab.dsp.resample import resample_audio
from respiralab.evaluation.metrics import LABEL_ORDER
from respiralab.models.baseline import load_split_features
from respiralab.models.traditional import build_pipeline

TARGET_SAMPLE_RATE = 4000
MAX_DURATION_S = 60.0
MODEL_ID = "E01_logistic_regression_frozen_cdc0b4f"
NOT_DIAGNOSTIC_WARNING = (
    "Research/educational tool only. Not a diagnostic device. "
    "This is a sound-event label, not a disease diagnosis."
)


class InputError(ValueError):
    """Raised for corrupt, malformed, or unsupported audio input."""


def validate_wav(audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio)
    if audio.size == 0:
        raise InputError("empty audio")
    if audio.ndim > 1:
        raise InputError("expected mono audio, got multi-channel")
    if not np.isfinite(audio).all():
        raise InputError("audio contains non-finite (NaN/inf) samples")
    if sample_rate <= 0:
        raise InputError(f"invalid sample rate: {sample_rate}")
    if len(audio) / sample_rate > MAX_DURATION_S:
        raise InputError(f"audio too long: exceeds {MAX_DURATION_S}s limit")
    if np.allclose(audio, 0.0):
        raise InputError("silent audio (all-zero)")


def build_frozen_predictor(feature_manifest_csv: str | Path, cache_dir: str | Path):
    manifest_df = pd.read_csv(feature_manifest_csv)
    X_train, y_train = load_split_features(manifest_df, "train", Path(cache_dir))
    pipeline = build_pipeline(seed=42)
    pipeline.fit(X_train, y_train)
    return pipeline


def _ordered_probabilities(pipeline, features: np.ndarray) -> list[float]:
    raw = pipeline.predict_proba(features)[0]
    by_class = dict(zip(pipeline.named_steps["clf"].classes_, raw, strict=True))
    return [float(by_class[label]) for label in LABEL_ORDER]


def analyze_cycle(pipeline, audio: np.ndarray, sample_rate: int) -> dict:
    validate_wav(audio, sample_rate)
    resampled = resample_audio(np.asarray(audio), sample_rate, TARGET_SAMPLE_RATE)
    features = extract_features(resampled, TARGET_SAMPLE_RATE).reshape(1, -1)
    return {
        "model_id": MODEL_ID,
        "label_names": list(LABEL_ORDER),
        "probabilities": _ordered_probabilities(pipeline, features),
        "warning": NOT_DIAGNOSTIC_WARNING,
        "signal_plot_data": {"waveform": resampled, "sample_rate": TARGET_SAMPLE_RATE},
        "analysis_mode": "annotated_cycle",
        "_debug_features": features[0].tolist(),
    }


def analyze_recording(
    pipeline, audio: np.ndarray, sample_rate: int, bounds: tuple[float, float] | None = None
) -> dict:
    audio = np.asarray(audio)
    if bounds is not None:
        start_s, end_s = bounds
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        if not (0 <= start_idx < end_idx <= len(audio)):
            raise InputError(f"invalid cycle bounds: start={start_s}, end={end_s}")
        segment = audio[start_idx:end_idx]
        mode = "annotated_cycle"
    else:
        segment = audio
        mode = "exploratory_window"

    result = analyze_cycle(pipeline, segment, sample_rate)
    result["analysis_mode"] = mode
    if mode == "exploratory_window":
        result["warning"] += (
            " Whole-recording input without expert cycle bounds is exploratory, "
            "not benchmark-equivalent."
        )
    return result

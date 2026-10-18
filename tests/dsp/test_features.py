import json
from pathlib import Path

import numpy as np

from respiralab.dsp.features import FEATURE_NAMES, extract_features


def _tone(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def test_feature_vector_has_fixed_length_matching_names():
    sr = 4000
    audio = _tone(300.0, 2.5, sr)
    vec = extract_features(audio, sr)
    assert len(vec) == len(FEATURE_NAMES)


def test_feature_vector_length_is_same_for_different_durations():
    sr = 4000
    short = extract_features(_tone(300.0, 0.5, sr), sr)
    long = extract_features(_tone(300.0, 8.0, sr), sr)
    assert len(short) == len(long) == len(FEATURE_NAMES)


def test_feature_vector_is_finite_for_valid_audio():
    sr = 4000
    vec = extract_features(_tone(500.0, 2.0, sr), sr)
    assert np.isfinite(vec).all()


def test_feature_vector_is_finite_for_silence():
    sr = 4000
    audio = np.zeros(sr * 2, dtype=np.float32)
    vec = extract_features(audio, sr)
    assert np.isfinite(vec).all()


def test_feature_vector_is_finite_for_very_short_audio():
    sr = 4000
    audio = _tone(300.0, 0.05, sr)
    vec = extract_features(audio, sr)
    assert np.isfinite(vec).all()
    assert len(vec) == len(FEATURE_NAMES)


def test_extraction_is_deterministic():
    sr = 4000
    audio = _tone(300.0, 2.0, sr)
    vec1 = extract_features(audio, sr)
    vec2 = extract_features(audio, sr)
    np.testing.assert_array_equal(vec1, vec2)


def test_zcr_distinguishes_high_vs_low_frequency():
    sr = 4000
    low = extract_features(_tone(50.0, 2.0, sr), sr)
    high = extract_features(_tone(800.0, 2.0, sr), sr)
    zcr_idx = FEATURE_NAMES.index("zcr_mean")
    assert high[zcr_idx] > low[zcr_idx]


def test_feature_schema_json_matches_feature_names():
    schema_path = Path(__file__).parent.parent.parent / "manifests" / "feature_schema.json"
    schema = json.loads(schema_path.read_text())
    assert schema["fields"] == FEATURE_NAMES
    assert schema["n_features"] == len(FEATURE_NAMES)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from respiralab.inference import v2_predict
from respiralab.models import ood_gate


def _tone(duration_s=1.0, sample_rate=4000, freq=300.0):
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    return 0.1 * np.sin(2 * np.pi * freq * t), sample_rate


def _write_fake_weights(path: Path) -> str:
    path.write_bytes(b"not real torch weights, only sha256 is checked before load")
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resolve_weights_accepts_matching_hash(tmp_path, monkeypatch):
    weights_path = tmp_path / "weights.pt"
    expected_sha256 = _write_fake_weights(weights_path)
    monkeypatch.setattr(v2_predict, "hf_hub_download", lambda **kwargs: str(weights_path))

    resolved = v2_predict.resolve_weights("org/repo", "weights.pt", expected_sha256)
    assert resolved == weights_path


def test_resolve_weights_refuses_mismatched_hash(tmp_path, monkeypatch):
    weights_path = tmp_path / "weights.pt"
    _write_fake_weights(weights_path)
    monkeypatch.setattr(v2_predict, "hf_hub_download", lambda **kwargs: str(weights_path))

    with pytest.raises(v2_predict.WeightsHashMismatchError, match="Refusing to load"):
        v2_predict.resolve_weights("org/repo", "weights.pt", expected_sha256="0" * 64)


class _FakeEncoder:
    def __call__(self, **inputs):
        class _Output:
            pooler_output = torch.zeros((1, 8))

        return _Output()


class _FakeModel:
    def __init__(self):
        self.encoder = _FakeEncoder()

    def classifier(self, pooled):
        return torch.tensor([[5.0, 0.0, 0.0, 0.0]])  # argmax -> "normal"


def _fake_predictor(threshold: float) -> dict:
    embeddings = np.zeros((20, 8), dtype=np.float32)
    gate = ood_gate.fit_gate(embeddings)
    gate["threshold"] = threshold  # override fitted threshold for deterministic accept/reject
    return {
        "model": _FakeModel(),
        "feature_extractor": lambda waveforms, sampling_rate, return_tensors: {
            "input_values": torch.zeros((1, 10))
        },
        "device": "cpu",
        "temperature": 1.0,
        "ood_gate": gate,
    }


def test_analyze_cycle_v2_accepts_in_distribution_input():
    predictor = _fake_predictor(threshold=1.0)  # fake embedding distance is 0.0
    audio, sr = _tone()

    result = v2_predict.analyze_cycle_v2(predictor, audio, sr)

    assert result["model_id"] == v2_predict.MODEL_ID
    assert result["label_names"][int(np.argmax(result["probabilities"]))] == "normal"
    assert result["ood_distance"] <= result["ood_threshold"]


def test_analyze_cycle_v2_rejects_out_of_distribution_input():
    predictor = _fake_predictor(threshold=-1.0)  # any real distance (>=0) now exceeds it
    audio, sr = _tone()

    with pytest.raises(v2_predict.OutOfDistributionError, match="refusing to predict"):
        v2_predict.analyze_cycle_v2(predictor, audio, sr)


def test_analyze_recording_v2_without_bounds_is_exploratory():
    predictor = _fake_predictor(threshold=1.0)
    audio, sr = _tone(duration_s=3.0)

    result = v2_predict.analyze_recording_v2(predictor, audio, sr, bounds=None)
    assert result["analysis_mode"] == "exploratory_window"
    assert "exploratory" in result["warning"].lower()


def test_analyze_recording_v2_with_bounds_is_annotated_cycle():
    predictor = _fake_predictor(threshold=1.0)
    audio, sr = _tone(duration_s=3.0)

    result = v2_predict.analyze_recording_v2(predictor, audio, sr, bounds=(0.5, 1.5))
    assert result["analysis_mode"] == "annotated_cycle"


def test_analyze_recording_v2_rejects_invalid_bounds():
    predictor = _fake_predictor(threshold=1.0)
    audio, sr = _tone(duration_s=1.0)

    with pytest.raises(v2_predict.InputError, match="bounds"):
        v2_predict.analyze_recording_v2(predictor, audio, sr, bounds=(0.9, 0.1))

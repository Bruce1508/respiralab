from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from respiralab.models import external_eval


def test_build_external_rows_parses_phases_and_counts_exclusions(tmp_path):
    train_dir = tmp_path / "HF_Lung_V1_train"
    train_dir.mkdir()
    # phase with crackle, phase with only rhonchi (excluded)
    (train_dir / "steth_a_label.txt").write_text(
        "I 00:00:00.000 00:00:01.000\n"
        "D 00:00:00.200 00:00:00.800\n"
        "E 00:00:02.000 00:00:03.000\n"
        "Rhonchi 00:00:02.200 00:00:02.800\n"
    )
    sf.write(str(train_dir / "steth_a.wav"), np.zeros(16000 * 3), 16000)

    rows, n_excluded = external_eval.build_external_rows({"train": train_dir})

    assert n_excluded == 1
    assert len(rows) == 1
    assert rows.iloc[0]["label4"] == "crackle"
    assert rows.iloc[0]["record_id"] == "HF_Lung_V1_train/steth_a"
    assert rows.iloc[0]["device"] == "steth"


def test_build_external_rows_skips_label_file_with_no_matching_wav(tmp_path):
    train_dir = tmp_path / "HF_Lung_V1_train"
    train_dir.mkdir()
    (train_dir / "trunc_b_label.txt").write_text("I 00:00:00.000 00:00:01.000\n")
    # no matching .wav written

    rows, n_excluded = external_eval.build_external_rows({"train": train_dir})

    assert len(rows) == 0
    assert n_excluded == 0


def test_stratified_sample_respects_requested_size_and_is_reproducible():
    rows = pd.DataFrame(
        {
            "label4": ["normal"] * 100 + ["crackle"] * 50 + ["wheeze"] * 30 + ["both"] * 5,
            "record_id": [f"r{i}" for i in range(185)],
        }
    )
    sample1 = external_eval.stratified_sample(rows, n=50, seed=42)
    sample2 = external_eval.stratified_sample(rows, n=50, seed=42)

    assert len(sample1) == len(sample2)
    assert set(sample1["record_id"]) == set(sample2["record_id"])
    assert len(sample1) < len(rows)
    # every class still represented proportionally, none dropped entirely
    assert set(sample1["label4"].unique()) == {"normal", "crackle", "wheeze", "both"}


def test_stratified_sample_returns_all_rows_when_n_exceeds_total():
    rows = pd.DataFrame({"label4": ["normal", "crackle"], "record_id": ["r0", "r1"]})
    sample = external_eval.stratified_sample(rows, n=1000, seed=42)
    assert len(sample) == 2


def test_evaluate_e01_external_predicts_and_skips_silent_segments(tmp_path, monkeypatch):
    audio_root = tmp_path
    rec_dir = audio_root / "split"
    rec_dir.mkdir()
    sr = 4000
    sf.write(str(rec_dir / "r0.wav"), np.random.default_rng(0).normal(size=sr * 2).astype(np.float32), sr)
    sf.write(str(rec_dir / "r1.wav"), np.zeros(sr * 2, dtype=np.float32), sr)  # silent -> skipped

    rows = pd.DataFrame(
        [
            {"record_id": "split/r0", "label4": "normal", "start_s": 0.0, "end_s": 1.0},
            {"record_id": "split/r1", "label4": "crackle", "start_s": 0.0, "end_s": 1.0},
        ]
    )

    class _FakePipeline:
        def predict(self, X):
            return np.array(["normal"] * len(X))

    result = external_eval.evaluate_e01_external(rows, audio_root, _FakePipeline())

    assert result["n_phases_evaluated"] == 1
    assert result["n_phases_skipped_invalid_audio"] == 1


def test_evaluate_e01_external_skips_degenerate_sub_millisecond_phases(tmp_path):
    """HF_Lung has a handful of ~1ms phases (annotation rounding); these must be
    skipped, not crash the whole run (librosa's spectral_contrast indexes out of
    bounds on near-empty arrays)."""
    audio_root = tmp_path
    rec_dir = audio_root / "split"
    rec_dir.mkdir()
    sr = 4000
    sf.write(str(rec_dir / "r0.wav"), np.random.default_rng(0).normal(size=sr * 2).astype(np.float32), sr)

    rows = pd.DataFrame(
        [
            {"record_id": "split/r0", "label4": "normal", "start_s": 0.0, "end_s": 1.0},
            {"record_id": "split/r0", "label4": "crackle", "start_s": 1.0, "end_s": 1.001},  # ~1ms
        ]
    )

    class _FakePipeline:
        def predict(self, X):
            return np.array(["normal"] * len(X))

    result = external_eval.evaluate_e01_external(rows, audio_root, _FakePipeline())

    assert result["n_phases_evaluated"] == 1
    assert result["n_phases_skipped_invalid_audio"] == 1


def test_evaluate_v2_external_computes_metrics_from_collected_logits(monkeypatch):
    rows_sampled = pd.DataFrame(
        {
            "record_id": ["split/r0", "split/r0", "split/r1", "split/r1"],
            "label4": ["normal", "crackle", "wheeze", "both"],
        }
    )
    fake_logits = np.array(
        [[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 5.0]]
    )
    fake_labels = np.array(["normal", "crackle", "wheeze", "both"])

    class _FakeModel:
        def load_state_dict(self, _state, strict=True):
            pass

        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(external_eval, "ASTClassifier", lambda unfreeze_last_n: _FakeModel())
    monkeypatch.setattr(external_eval.torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(external_eval, "collect_logits", lambda model, loader, device: (fake_logits, fake_labels))

    result = external_eval.evaluate_v2_external(
        rows_sampled, audio_root=Path("unused"), weights_path=Path("unused.pt"), device="cpu", feature_extractor=None
    )

    assert result["n_phases_evaluated"] == 4
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["sample_size_requested"] == external_eval.V2_SAMPLE_SIZE

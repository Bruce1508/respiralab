from pathlib import Path

import numpy as np

from respiralab.dsp.feature_cache import (
    CacheRow,
    cache_one_cycle,
    feature_vector_hash,
    preprocess_config_hash,
    write_manifest,
)


def _tone(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def test_feature_vector_hash_is_deterministic():
    vec = np.array([1.0, 2.0, 3.0])
    assert feature_vector_hash(vec) == feature_vector_hash(vec.copy())


def test_feature_vector_hash_differs_for_different_vectors():
    assert feature_vector_hash(np.array([1.0, 2.0])) != feature_vector_hash(np.array([1.0, 2.1]))


def test_preprocess_config_hash_is_stable_across_calls():
    assert preprocess_config_hash() == preprocess_config_hash()


def test_cache_one_cycle_writes_npy_and_returns_row(tmp_path: Path):
    audio = _tone(300.0, 2.0, 4000)
    row = cache_one_cycle(
        audio=audio,
        sample_rate=4000,
        cycle_id="101_1b1_Al_sc_Meditron_c000",
        record_id="101_1b1_Al_sc_Meditron",
        patient_id="101",
        split="train",
        label4="crackle",
        input_sha256="deadbeef",
        cache_dir=tmp_path,
    )
    assert isinstance(row, CacheRow)
    cached_path = tmp_path / row.cache_path
    assert cached_path.exists()
    loaded = np.load(cached_path)
    assert len(loaded) == 46


def test_identical_audio_produces_identical_feature_hash(tmp_path: Path):
    audio = _tone(300.0, 2.0, 4000)
    row1 = cache_one_cycle(
        audio=audio,
        sample_rate=4000,
        cycle_id="c1",
        record_id="r1",
        patient_id="p1",
        split="train",
        label4="normal",
        input_sha256="abc",
        cache_dir=tmp_path / "a",
    )
    row2 = cache_one_cycle(
        audio=audio.copy(),
        sample_rate=4000,
        cycle_id="c2",
        record_id="r1",
        patient_id="p1",
        split="train",
        label4="normal",
        input_sha256="abc",
        cache_dir=tmp_path / "b",
    )
    assert row1.feature_hash == row2.feature_hash


def test_feature_hash_is_independent_of_split_label_no_mixing(tmp_path: Path):
    audio = _tone(500.0, 2.0, 4000)
    row_train = cache_one_cycle(
        audio=audio,
        sample_rate=4000,
        cycle_id="c1",
        record_id="r1",
        patient_id="p1",
        split="train",
        label4="wheeze",
        input_sha256="xyz",
        cache_dir=tmp_path / "t",
    )
    row_test = cache_one_cycle(
        audio=audio.copy(),
        sample_rate=4000,
        cycle_id="c1",
        record_id="r1",
        patient_id="p1",
        split="test",
        label4="wheeze",
        input_sha256="xyz",
        cache_dir=tmp_path / "v",
    )
    assert row_train.feature_hash == row_test.feature_hash


def test_write_manifest_has_expected_columns(tmp_path: Path):
    audio = _tone(300.0, 1.0, 4000)
    row = cache_one_cycle(
        audio=audio,
        sample_rate=4000,
        cycle_id="c1",
        record_id="r1",
        patient_id="p1",
        split="val",
        label4="both",
        input_sha256="hash123",
        cache_dir=tmp_path,
    )
    out = tmp_path / "feature_manifest.csv"
    write_manifest([row], out)
    import csv

    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "cycle_id",
            "record_id",
            "patient_id",
            "split",
            "label4",
            "input_sha256",
            "preprocess_config_hash",
            "feature_hash",
            "cache_path",
        ]
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["split"] == "val"

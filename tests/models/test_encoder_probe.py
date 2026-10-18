from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from respiralab.models import encoder_probe
from respiralab.models.encoder_probe import build_trainval_rows, extract_and_cache_embeddings


def _manifest_and_cycles():
    manifest_df = pd.DataFrame(
        [
            {"cycle_id": "r1_c0", "patient_id": "p1", "split": "train", "label4": "normal"},
            {"cycle_id": "r1_c1", "patient_id": "p1", "split": "train", "label4": "crackle"},
            {"cycle_id": "r2_c0", "patient_id": "p2", "split": "val", "label4": "wheeze"},
            {"cycle_id": "r3_c0", "patient_id": "p3", "split": "test", "label4": "both"},
        ]
    )
    cycles_df = pd.DataFrame(
        [
            {"cycle_id": "r1_c0", "record_id": "r1", "start_s": 0.0, "end_s": 1.0},
            {"cycle_id": "r1_c1", "record_id": "r1", "start_s": 1.0, "end_s": 2.0},
            {"cycle_id": "r2_c0", "record_id": "r2", "start_s": 0.0, "end_s": 1.0},
            {"cycle_id": "r3_c0", "record_id": "r3", "start_s": 0.0, "end_s": 1.0},
        ]
    )
    return manifest_df, cycles_df


def test_build_trainval_rows_excludes_test_and_adds_bounds():
    manifest_df, cycles_df = _manifest_and_cycles()
    rows = build_trainval_rows(manifest_df, cycles_df)
    assert set(rows["cycle_id"]) == {"r1_c0", "r1_c1", "r2_c0"}
    assert list(rows["start_s"]) == [0.0, 1.0, 0.0]
    assert list(rows["record_id"]) == ["r1", "r1", "r2"]


def test_build_trainval_rows_raises_on_missing_cycle_bounds():
    manifest_df, cycles_df = _manifest_and_cycles()
    cycles_df = cycles_df[cycles_df["cycle_id"] != "r1_c0"]  # drop bounds for a train row
    with pytest.raises(ValueError, match="missing from cycles.csv"):
        build_trainval_rows(manifest_df, cycles_df)


def test_extract_and_cache_embeddings_caches_and_is_resumable(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    cache_dir = tmp_path / "cache"
    sr = 4000
    for record_id in ("r1", "r2"):
        sf.write(str(audio_dir / f"{record_id}.wav"), np.zeros(sr * 3, dtype=np.float32), sr)

    manifest_df, cycles_df = _manifest_and_cycles()
    rows = build_trainval_rows(manifest_df, cycles_df)

    calls = {"n": 0}

    def fake_embed_batch(audio_batch, sample_rate):
        calls["n"] += 1
        return np.stack([np.full(4, float(len(a))) for a in audio_batch])

    monkeypatch.setattr(encoder_probe, "embed_batch", fake_embed_batch)
    monkeypatch.setattr(encoder_probe, "EMBEDDING_DIM", 4)

    emb1 = extract_and_cache_embeddings(rows, audio_dir, cache_dir)
    assert emb1.shape == (3, 4)
    assert calls["n"] == 1  # one batch covering all 3 uncached rows
    assert (cache_dir / "r1_c0.npy").exists()

    # second call: everything is cached, embed_batch must not be invoked again
    emb2 = extract_and_cache_embeddings(rows, audio_dir, cache_dir)
    np.testing.assert_array_equal(emb1, emb2)
    assert calls["n"] == 1

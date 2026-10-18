from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from respiralab.models.dataset import (
    LABEL_TO_INDEX,
    LogMelCycleDataset,
    build_frame_mask,
    pad_or_crop,
)


def test_pad_or_crop_pads_short_audio_with_zeros_and_reports_real_length():
    audio = np.full(100, 7.0)
    fixed, n_real = pad_or_crop(audio, target_n_samples=1000)
    assert fixed.shape == (1000,)
    assert np.all(fixed[:100] == 7.0)
    assert np.all(fixed[100:] == 0.0)
    assert n_real == 100


def test_pad_or_crop_crops_long_audio_to_target_length():
    audio = np.arange(2000, dtype=np.float64)
    fixed, n_real = pad_or_crop(audio, target_n_samples=1000)
    assert fixed.shape == (1000,)
    assert np.array_equal(fixed, audio[:1000])
    assert n_real == 1000


def test_build_frame_mask_marks_only_real_frames_true():
    mask = build_frame_mask(n_real_samples=320, n_frames=20, hop_length=64)
    assert mask.shape == (20,)
    assert mask[:5].all()
    assert not mask[5:].any()


def test_build_frame_mask_never_exceeds_total_frames():
    mask = build_frame_mask(n_real_samples=100_000, n_frames=10, hop_length=64)
    assert mask.shape == (10,)
    assert mask.all()


def _write_wav(path, duration_s: float, sample_rate: int) -> None:
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    tone = 0.1 * np.sin(2 * np.pi * 300.0 * t)
    sf.write(str(path), tone, sample_rate)


def test_dataset_produces_identical_fixed_shape_for_variable_duration_and_native_rate(
    tmp_path,
):
    _write_wav(tmp_path / "rec_short.wav", duration_s=0.05, sample_rate=4000)
    _write_wav(tmp_path / "rec_long.wav", duration_s=5.0, sample_rate=44100)

    cycles_df = pd.DataFrame(
        [
            {
                "cycle_id": "c_short",
                "record_id": "rec_short",
                "start_s": 0.0,
                "end_s": 0.05,
                "label4": "normal",
            },
            {
                "cycle_id": "c_long",
                "record_id": "rec_long",
                "start_s": 0.0,
                "end_s": 5.0,
                "label4": "crackle",
            },
        ]
    )

    dataset = LogMelCycleDataset(
        cycles_df, tmp_path, target_sample_rate=4000, target_n_samples=400
    )

    assert len(dataset) == 2
    item_short = dataset[0]
    item_long = dataset[1]

    assert item_short["log_mel"].shape == item_long["log_mel"].shape
    assert item_short["mask"].shape == item_long["mask"].shape

    assert item_short["mask"].any() and not item_short["mask"].all()
    assert item_long["mask"].all()

    assert item_short["label"].item() == LABEL_TO_INDEX["normal"]
    assert item_long["label"].item() == LABEL_TO_INDEX["crackle"]
    assert item_short["cycle_id"] == "c_short"


def test_dataset_applies_augment_fn_only_to_flagged_rows(tmp_path):
    from respiralab.models.augment import make_augment_fn

    _write_wav(tmp_path / "rec.wav", duration_s=1.0, sample_rate=4000)
    cycles_df = pd.DataFrame(
        [
            {
                "cycle_id": "c_plain",
                "record_id": "rec",
                "start_s": 0.0,
                "end_s": 1.0,
                "label4": "normal",
                "is_augmented": False,
                "aug_seed": 0,
            },
            {
                "cycle_id": "c_aug",
                "record_id": "rec",
                "start_s": 0.0,
                "end_s": 1.0,
                "label4": "normal",
                "is_augmented": True,
                "aug_seed": 1,
            },
        ]
    )

    plain_dataset = LogMelCycleDataset(cycles_df, tmp_path, target_sample_rate=4000, target_n_samples=400)
    augmented_dataset = LogMelCycleDataset(
        cycles_df, tmp_path, target_sample_rate=4000, target_n_samples=400, augment_fn=make_augment_fn()
    )

    plain_item = plain_dataset[0]
    augment_hook_on_unflagged_row = augmented_dataset[0]
    augmented_item = augmented_dataset[1]

    assert torch.equal(plain_item["log_mel"], augment_hook_on_unflagged_row["log_mel"])
    assert not torch.equal(plain_item["log_mel"], augmented_item["log_mel"])


def test_dataset_raises_on_unknown_label(tmp_path):
    _write_wav(tmp_path / "rec.wav", duration_s=0.5, sample_rate=4000)
    cycles_df = pd.DataFrame(
        [{"cycle_id": "c1", "record_id": "rec", "start_s": 0.0, "end_s": 0.5, "label4": "nope"}]
    )
    dataset = LogMelCycleDataset(cycles_df, tmp_path, target_sample_rate=4000, target_n_samples=400)
    with pytest.raises(KeyError):
        dataset[0]

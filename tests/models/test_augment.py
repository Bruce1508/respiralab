from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from respiralab.models.augment import augment_train_split, augment_waveform, build_augmented_cycles


def _cycles_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cycle_id": "c1", "record_id": "r1", "patient_id": 1, "start_s": 0.0, "end_s": 1.0, "label4": "normal"},
            {"cycle_id": "c2", "record_id": "r1", "patient_id": 1, "start_s": 1.0, "end_s": 2.0, "label4": "crackle"},
        ]
    )


def test_augment_waveform_is_seed_reproducible_and_preserves_length():
    audio = np.linspace(-1.0, 1.0, 400).astype(np.float64)
    out_a = augment_waveform(audio, np.random.default_rng(7))
    out_b = augment_waveform(audio, np.random.default_rng(7))
    assert out_a.shape == audio.shape
    assert np.array_equal(out_a, out_b)
    assert not np.array_equal(out_a, audio)


def test_build_augmented_cycles_inherits_parent_patient_and_gets_new_cycle_id():
    cycles_df = _cycles_df()
    augmented = build_augmented_cycles(
        cycles_df, train_patient_ids={1}, n_per_cycle=2, seed=0
    )
    assert len(augmented) == 4
    assert set(augmented["parent_cycle_id"]) == {"c1", "c2"}
    assert augmented["cycle_id"].is_unique
    assert (augmented["patient_id"] == 1).all()
    assert set(augmented["label4"]) == {"normal", "crackle"}
    assert (augmented["is_augmented"]).all()


def test_build_augmented_cycles_rejects_patient_outside_train_set():
    cycles_df = _cycles_df()
    with pytest.raises(ValueError, match="patient"):
        build_augmented_cycles(cycles_df, train_patient_ids={999}, n_per_cycle=1, seed=0)


def test_no_augmented_child_assigned_to_validation_or_test():
    cycles_df = pd.DataFrame(
        [
            {"cycle_id": "c1", "record_id": "r1", "patient_id": 1, "start_s": 0.0, "end_s": 1.0, "label4": "normal"},
            {"cycle_id": "c2", "record_id": "r2", "patient_id": 2, "start_s": 0.0, "end_s": 1.0, "label4": "wheeze"},
            {"cycle_id": "c3", "record_id": "r3", "patient_id": 3, "start_s": 0.0, "end_s": 1.0, "label4": "normal"},
            {"cycle_id": "c4", "record_id": "r4", "patient_id": 4, "start_s": 0.0, "end_s": 1.0, "label4": "both"},
        ]
    )
    splits_df = pd.DataFrame(
        [
            {"record_id": "r1", "patient_id": 1, "split": "train"},
            {"record_id": "r2", "patient_id": 2, "split": "train"},
            {"record_id": "r3", "patient_id": 3, "split": "val"},
            {"record_id": "r4", "patient_id": 4, "split": "test"},
        ]
    )
    augmented = augment_train_split(cycles_df, splits_df, n_per_cycle=3, seed=0)

    val_test_patients = {3, 4}
    assert not augmented["patient_id"].isin(val_test_patients).any()
    assert set(augmented["patient_id"]) == {1, 2}

from __future__ import annotations

import numpy as np
import pandas as pd


def augment_waveform(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One of {additive noise, circular time-shift, gain} chosen by rng."""
    choice = rng.integers(0, 3)
    if choice == 0:
        noise = rng.normal(0.0, 0.01, size=audio.shape)
        return audio + noise
    if choice == 1:
        shift = int(rng.integers(0, max(len(audio), 1)))
        return np.roll(audio, shift)
    gain = rng.uniform(0.7, 1.3)
    return audio * gain


def build_augmented_cycles(
    cycles_df: pd.DataFrame, train_patient_ids: set, n_per_cycle: int, seed: int
) -> pd.DataFrame:
    """Build augmented child rows for cycles_df, inheriting parent patient/record/label.

    Raises if any row's patient is outside train_patient_ids - augmentation must
    only ever be applied to the training split, never to val/test.
    """
    bad_patients = set(cycles_df["patient_id"]) - set(train_patient_ids)
    if bad_patients:
        raise ValueError(f"patient ids outside train_patient_ids: {bad_patients}")

    rows = []
    for _, row in cycles_df.iterrows():
        for k in range(n_per_cycle):
            child = row.to_dict()
            child["parent_cycle_id"] = row["cycle_id"]
            child["cycle_id"] = f"{row['cycle_id']}__aug{k}"
            child["is_augmented"] = True
            child["aug_seed"] = seed * 100_003 + hash((row["cycle_id"], k)) % 100_000
            rows.append(child)
    return pd.DataFrame(rows)


def augment_train_split(
    cycles_df: pd.DataFrame, splits_df: pd.DataFrame, n_per_cycle: int, seed: int
) -> pd.DataFrame:
    """Augment only the cycles whose record belongs to the train split."""
    merged = cycles_df.merge(splits_df[["record_id", "split"]], on="record_id")
    train_rows = merged[merged["split"] == "train"].drop(columns=["split"])
    train_patient_ids = set(train_rows["patient_id"])
    return build_augmented_cycles(train_rows, train_patient_ids, n_per_cycle, seed)


def make_augment_fn():
    def _augment_fn(audio: np.ndarray, row: pd.Series) -> np.ndarray:
        rng = np.random.default_rng(int(row["aug_seed"]))
        return augment_waveform(audio, rng)

    return _augment_fn

import pandas as pd
import pytest

from respiralab.data.split import (
    assert_patient_disjoint,
    build_report,
    make_primary_split,
    summarize_split,
)


def _synthetic_data(n_patients: int = 40, cycles_per_patient: int = 10):
    import random

    rng = random.Random(42)
    inventory_rows = []
    cycle_rows = []
    labels_cycle = ["normal", "crackle", "wheeze", "both"]
    for i in range(n_patients):
        patient_id = str(100 + i)
        dominant = labels_cycle[i % 4]
        record_id = f"{patient_id}_1b1_Al_sc_Meditron"
        inventory_rows.append({"record_id": record_id, "patient_id": patient_id})
        for c in range(cycles_per_patient):
            label = dominant if rng.random() < 0.7 else rng.choice(labels_cycle)
            cycle_rows.append({"record_id": record_id, "patient_id": patient_id, "label4": label})
    return (pd.DataFrame(inventory_rows), pd.DataFrame(cycle_rows))


def test_split_is_patient_disjoint():
    inventory_df, cycles_df = _synthetic_data()
    split_df = make_primary_split(inventory_df, cycles_df)
    assert_patient_disjoint(split_df)


def test_every_patient_assigned_exactly_one_split():
    inventory_df, cycles_df = _synthetic_data()
    split_df = make_primary_split(inventory_df, cycles_df)
    assert split_df["split"].isna().sum() == 0
    assert set(split_df["split"].unique()) <= {"train", "val", "test"}


def test_split_is_deterministic_given_same_seed():
    inventory_df, cycles_df = _synthetic_data()
    split1 = make_primary_split(inventory_df, cycles_df, seed=42)
    split2 = make_primary_split(inventory_df, cycles_df, seed=42)
    pd.testing.assert_frame_equal(
        split1.sort_values("record_id").reset_index(drop=True),
        split2.sort_values("record_id").reset_index(drop=True),
    )


def test_different_seed_can_change_assignment():
    inventory_df, cycles_df = _synthetic_data()
    split1 = make_primary_split(inventory_df, cycles_df, seed=42)
    split2 = make_primary_split(inventory_df, cycles_df, seed=13)
    assert_patient_disjoint(split1)
    assert_patient_disjoint(split2)


def test_leakage_check_raises_on_manually_corrupted_split():
    inventory_df, cycles_df = _synthetic_data()
    split_df = make_primary_split(inventory_df, cycles_df)
    corrupted = split_df.copy()
    leaked_patient_rows = corrupted[corrupted.split == "train"].head(1).copy()
    leaked_patient_rows["split"] = "test"
    leaked_patient_rows["record_id"] = leaked_patient_rows["record_id"] + "_dup"
    corrupted = pd.concat([corrupted, leaked_patient_rows], ignore_index=True)
    with pytest.raises(AssertionError):
        assert_patient_disjoint(corrupted)


def test_ratios_are_roughly_70_15_15():
    inventory_df, cycles_df = _synthetic_data(n_patients=100, cycles_per_patient=5)
    split_df = make_primary_split(inventory_df, cycles_df)
    counts = split_df.groupby("split")["patient_id"].nunique()
    total = counts.sum()
    assert 0.55 <= counts.get("train", 0) / total <= 0.85
    assert 0.05 <= counts.get("val", 0) / total <= 0.3
    assert 0.05 <= counts.get("test", 0) / total <= 0.3


def test_report_flags_missing_class_in_a_split():
    summaries_input = pd.DataFrame(
        {
            "record_id": ["101_1b1_Al_sc_Meditron", "102_1b1_Ar_sc_Meditron"],
            "patient_id": ["101", "102"],
            "split": ["train", "test"],
        }
    )
    cycles_df = pd.DataFrame(
        {
            "record_id": ["101_1b1_Al_sc_Meditron", "102_1b1_Ar_sc_Meditron"],
            "patient_id": ["101", "102"],
            "label4": ["wheeze", "normal"],
        }
    )
    summaries = summarize_split(summaries_input, cycles_df)
    report = build_report(summaries)
    assert "ZERO cycles of class" in report


def test_report_states_full_coverage_when_all_classes_present_everywhere():
    inventory_df, cycles_df = _synthetic_data(n_patients=100, cycles_per_patient=20)
    split_df = make_primary_split(inventory_df, cycles_df)
    summaries = summarize_split(split_df, cycles_df)
    report = build_report(summaries)
    assert "All 4 classes present" in report or "ZERO cycles of class" in report

"""D07 (Gate G1): this project's own reproducible, patient-disjoint primary
train/val/test split.

Splits at the PATIENT level (never a recording or cycle) via sklearn's
StratifiedKFold, stratifying on each patient's dominant cycle label4 as a
best-effort class-balance signal — patients rarely have one uniform label,
so this is "attempts at class balance" per
docs/03_DATA_AND_EVALUATION_CONTRACT.md, not a hard guarantee. Target ratios
(~70/15/15) are guidance, not an exact constraint. Seed fixed at 42
(docs/04_EXPERIMENTS.md). This is independent of, and never combined with,
the official ICBHI split audited in D06.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

SPLIT_SEED = 42
# n_splits chosen <= the smallest dominant-label class size at each stage so
# every class can appear in every fold; see split_audit report for the
# resulting exact per-split class coverage (not guaranteed, only attempted).
TEST_N_SPLITS = 6  # 1 fold ~16.7% test
VAL_N_SPLITS = 5  # 1 fold ~20% of the train+val remainder (~17% of total)


def patient_dominant_label(cycles_df: pd.DataFrame) -> pd.Series:
    """Each patient's most frequent cycle label4 — the stratification signal."""
    labels = cycles_df.copy()
    labels["patient_id"] = labels["patient_id"].astype(str)
    return labels.groupby("patient_id")["label4"].agg(lambda s: s.value_counts().idxmax())


def make_primary_split(
    inventory_df: pd.DataFrame,
    cycles_df: pd.DataFrame,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Return one row per recording: record_id, patient_id, split (train/val/test).

    Patient-disjoint by construction: splitting happens on one-row-per-patient
    arrays (via StratifiedKFold on the dominant-label signal), then every
    recording of a patient is mapped to that patient's assigned split — no
    recording-level or cycle-level split ever occurs.
    """
    dominant = patient_dominant_label(cycles_df)
    patients = dominant.index.to_numpy()
    y = dominant.to_numpy()

    test_kfold = StratifiedKFold(n_splits=TEST_N_SPLITS, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(test_kfold.split(patients, y))
    trainval_patients, trainval_y = patients[trainval_idx], y[trainval_idx]
    test_patients = set(patients[test_idx])

    val_kfold = StratifiedKFold(n_splits=VAL_N_SPLITS, shuffle=True, random_state=seed)
    train_idx, val_idx = next(val_kfold.split(trainval_patients, trainval_y))
    train_patients = set(trainval_patients[train_idx])
    val_patients = set(trainval_patients[val_idx])

    patient_to_split: dict[str, str] = {}
    patient_to_split.update({p: "train" for p in train_patients})
    patient_to_split.update({p: "val" for p in val_patients})
    patient_to_split.update({p: "test" for p in test_patients})

    out = inventory_df[["record_id", "patient_id"]].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["split"] = out["patient_id"].map(patient_to_split)
    return out


def assert_patient_disjoint(split_df: pd.DataFrame) -> None:
    """The leakage control tests from docs/03_DATA_AND_EVALUATION_CONTRACT.md."""
    train_patients = set(split_df.loc[split_df.split == "train", "patient_id"])
    val_patients = set(split_df.loc[split_df.split == "val", "patient_id"])
    test_patients = set(split_df.loc[split_df.split == "test", "patient_id"])
    train_recordings = set(split_df.loc[split_df.split == "train", "record_id"])
    test_recordings = set(split_df.loc[split_df.split == "test", "record_id"])

    assert not (train_patients & val_patients), "train/val patient overlap"
    assert not (train_patients & test_patients), "train/test patient overlap"
    assert not (val_patients & test_patients), "val/test patient overlap"
    assert train_recordings.isdisjoint(test_recordings), "train/test recording overlap"


@dataclass
class SplitSummary:
    split: str
    n_patients: int
    n_recordings: int
    n_cycles: int
    label_counts: dict[str, int]


def summarize_split(split_df: pd.DataFrame, cycles_df: pd.DataFrame) -> list[SplitSummary]:
    cycles = cycles_df.copy()
    cycles["patient_id"] = cycles["patient_id"].astype(str)
    record_to_split = dict(zip(split_df["record_id"], split_df["split"], strict=True))
    cycles["split"] = cycles["record_id"].map(record_to_split)

    summaries = []
    for split in ("train", "val", "test"):
        sub = split_df[split_df.split == split]
        csub = cycles[cycles.split == split]
        label_counts = csub["label4"].value_counts().to_dict()
        summaries.append(
            SplitSummary(
                split=split,
                n_patients=sub["patient_id"].nunique(),
                n_recordings=len(sub),
                n_cycles=len(csub),
                label_counts={label: label_counts.get(label, 0) for label in LABEL_ORDER},
            )
        )
    return summaries


LABEL_ORDER = ("normal", "crackle", "wheeze", "both")


def build_report(summaries: list[SplitSummary]) -> str:
    missing_classes = [
        (s.split, label)
        for s in summaries
        for label in LABEL_ORDER
        if s.label_counts.get(label, 0) == 0
    ]
    coverage_section = (
        "All 4 classes present with at least 1 cycle in every split."
        if not missing_classes
        else "\n".join(
            f"- **{split}** has ZERO cycles of class **{label}** — rare-class coverage gap, "
            "see docs/03_DATA_AND_EVALUATION_CONTRACT.md fallback guidance "
            "(group-stratified CV for development) before relying on this split for that class."
            for split, label in missing_classes
        )
    )
    rows = "\n".join(
        f"| {s.split} | {s.n_patients} | {s.n_recordings} | {s.n_cycles} | "
        + " | ".join(str(s.label_counts.get(label, 0)) for label in LABEL_ORDER)
        + " |"
        for s in summaries
    )
    total_patients = sum(s.n_patients for s in summaries)
    return f"""# Primary patient-disjoint split audit (D07, Gate G1)

Seed: {SPLIT_SEED}. Patient-level StratifiedKFold on each patient's dominant
cycle label (test: {TEST_N_SPLITS} folds, val: {VAL_N_SPLITS} folds of the
remainder). Ratios are guidance (~70/15/15), not an exact target.

## Leakage checks
Zero patient overlap across train/val/test, zero recording overlap between
train and test (asserted in `tests/test_splits.py`, re-verified here: all
{total_patients} patients land in exactly one split).

## Rare-class coverage
{coverage_section}

## Per-split counts
| split | patients | recordings | cycles | normal | crackle | wheeze | both |
|---|---|---|---|---|---|---|---|
{rows}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the D07 primary patient-disjoint split.")
    parser.add_argument("--dataset-inventory", required=True, type=Path)
    parser.add_argument("--cycles", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    args = parser.parse_args(argv)

    inventory_df = pd.read_csv(args.dataset_inventory)
    cycles_df = pd.read_csv(args.cycles)

    split_df = make_primary_split(inventory_df, cycles_df)
    assert_patient_disjoint(split_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.out, index=False)

    summaries = summarize_split(split_df, cycles_df)
    report = build_report(summaries)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report)

    print("leakage checks: PASS (zero patient/recording overlap)")
    for s in summaries:
        print(f"{s.split}: {s.n_patients} patients, {s.n_recordings} recordings, "
              f"{s.n_cycles} cycles, labels={s.label_counts}")
    print(f"wrote split: {args.out}")
    print(f"wrote report: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

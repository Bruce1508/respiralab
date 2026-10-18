"""D44: 5-fold patient-grouped stratified CV over train+val (ADR-0001 decision D3).

Pools the frozen primary split's train+val patients only -- the test split's
patients never enter a fold -- and reports CV mean/sd instead of a single
noisy ~21-patient val split. See docs/adr/0001-v2-pretrained-encoder-and-evaluation.md.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from respiralab.data.inventory import sha256_of_file
from respiralab.evaluation.metrics import score_predictions
from respiralab.models.traditional import build_pipeline

CV_SEED = 42
N_SPLITS = 5


def load_trainval_pool(
    manifest_df: pd.DataFrame, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cycle-level features/labels/patient_ids for train+val only; test is excluded."""
    rows = manifest_df[manifest_df["split"].isin(("train", "val"))]
    if rows.empty:
        raise ValueError("no feature_manifest rows for split in {train, val}")
    cache_dir = Path(cache_dir)
    X = np.stack([np.load(cache_dir / p) for p in rows["cache_path"]])
    y = rows["label4"].to_numpy()
    patient_ids = rows["patient_id"].astype(str).to_numpy()
    return X, y, patient_ids


def make_folds(
    patient_ids: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS, seed: int = CV_SEED
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Patient-grouped, label-stratified fold indices over the cycle-level arrays.

    StratifiedGroupKFold keeps every patient's cycles in exactly one fold
    (hard group constraint) while trying to preserve per-fold class balance.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups=patient_ids))


def assert_no_patient_crosses_folds(
    patient_ids: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    val_patient_sets = [set(patient_ids[val_idx]) for _, val_idx in folds]
    for i, pset_i in enumerate(val_patient_sets):
        for j, pset_j in enumerate(val_patient_sets):
            if i != j:
                assert not pset_i & pset_j, f"patient(s) {pset_i & pset_j} in both fold {i} and {j}"


def run_grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    build_fn: Callable[[], Pipeline],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Fit build_fn() fresh per fold on the fold's own train split; score on held-out fold."""
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        pipeline = build_fn()
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[val_idx])
        scores = score_predictions(y[val_idx], y_pred)
        fold_results.append(
            {
                "fold": fold_idx,
                "n_train_patients": len(set(patient_ids[train_idx])),
                "n_val_patients": len(set(patient_ids[val_idx])),
                "n_val_cycles": len(val_idx),
                "macro_f1": scores["macro_f1"],
                "per_class": scores["per_class"],
            }
        )
    macro_f1s = np.array([f["macro_f1"] for f in fold_results])
    per_class_recalls: dict[str, list[float]] = {}
    for f in fold_results:
        for label, stats in f["per_class"].items():
            per_class_recalls.setdefault(label, []).append(stats["recall"])
    return {
        "n_splits": len(fold_results),
        "folds": fold_results,
        "macro_f1_mean": float(macro_f1s.mean()),
        "macro_f1_sd": float(macro_f1s.std(ddof=1)),
        "per_class_recall_mean": {k: float(np.mean(v)) for k, v in per_class_recalls.items()},
        "per_class_recall_sd": {k: float(np.std(v, ddof=1)) for k, v in per_class_recalls.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D44: 5-fold patient-grouped CV over train+val; E01 as v2 reference."
    )
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=CV_SEED)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    X, y, patient_ids = load_trainval_pool(manifest_df, args.cache_dir)
    folds = make_folds(patient_ids, y, n_splits=args.n_splits, seed=args.seed)
    assert_no_patient_crosses_folds(patient_ids, folds)

    cv_result = run_grouped_cv(X, y, patient_ids, build_pipeline, folds)
    result = {
        "protocol": "5-fold patient-grouped stratified CV over train+val (ADR-0001 D3); test excluded",
        "split_hash": sha256_of_file(args.splits_csv),
        "n_trainval_patients": len(set(patient_ids)),
        "n_trainval_cycles": len(y),
        "seed": args.seed,
        "model": "logistic_regression (E01 pipeline, v2 reference)",
        **cv_result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "cv_scores.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(f"E01 grouped-CV macro_f1: {result['macro_f1_mean']:.4f} +/- {result['macro_f1_sd']:.4f}")
    for label, mean in result["per_class_recall_mean"].items():
        sd = result["per_class_recall_sd"][label]
        print(f"  {label} recall: {mean:.4f} +/- {sd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

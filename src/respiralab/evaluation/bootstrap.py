from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from respiralab.evaluation.metrics import score_predictions
from respiralab.models.baseline import load_split_features, load_split_patient_ids
from respiralab.models.svm_search import build_svm_pipeline
from respiralab.models.traditional import build_pipeline

EVAL_BOOTSTRAP_SEED = 42
N_REPLICATES = 1000
TRAIN_SEEDS = (13, 21, 42)
BEST_SVM_PARAMS = {"C": 1.0, "gamma": 0.01}


def macro_f1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return score_predictions(y_true, y_pred)["macro_f1"]


def patient_bootstrap_ci(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_replicates: int = N_REPLICATES,
    seed: int = EVAL_BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> dict:
    patient_ids = np.asarray(patient_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    indices_by_patient = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    rng = np.random.default_rng(seed)
    boot_scores = np.empty(n_replicates)
    for i in range(n_replicates):
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        idx = np.concatenate([indices_by_patient[p] for p in sampled_patients])
        boot_scores[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1 - ci) / 2
    return {
        "point_estimate": float(metric_fn(y_true, y_pred)),
        "ci_lower": float(np.quantile(boot_scores, alpha)),
        "ci_upper": float(np.quantile(boot_scores, 1 - alpha)),
        "ci_level": ci,
        "n_replicates": n_replicates,
        "n_patients": n_patients,
        "seed": seed,
    }


MODEL_BUILDERS = {
    "logistic_regression": lambda seed: build_pipeline(seed=seed),
    "svm_rbf_best": lambda seed: build_svm_pipeline(seed=seed, **BEST_SVM_PARAMS),
}


def run_seed_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    patient_ids_dev: np.ndarray,
) -> list[dict]:
    rows = []
    for model_name, build_fn in MODEL_BUILDERS.items():
        for train_seed in TRAIN_SEEDS:
            pipeline = build_fn(train_seed)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_dev)
            ci_result = patient_bootstrap_ci(patient_ids_dev, y_dev, y_pred, macro_f1_metric)
            rows.append(
                {
                    "model": model_name,
                    "train_seed": train_seed,
                    "macro_f1": ci_result["point_estimate"],
                    "ci_lower": ci_result["ci_lower"],
                    "ci_upper": ci_result["ci_upper"],
                    "ci_level": ci_result["ci_level"],
                    "n_patients_val": ci_result["n_patients"],
                    "n_bootstrap_replicates": ci_result["n_replicates"],
                    "bootstrap_seed": ci_result["seed"],
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D20: patient-level bootstrap CI + 3-train-seed variability."
    )
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    X_train, y_train = load_split_features(manifest_df, "train", args.cache_dir)
    X_dev, y_dev = load_split_features(manifest_df, "val", args.cache_dir)
    patient_ids_dev = load_split_patient_ids(manifest_df, "val")

    rows = run_seed_comparison(X_train, y_train, X_dev, y_dev, patient_ids_dev)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}")
    for row in rows:
        print(
            f"{row['model']} seed={row['train_seed']}: macro_f1={row['macro_f1']:.4f} "
            f"CI=[{row['ci_lower']:.4f},{row['ci_upper']:.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

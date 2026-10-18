from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from respiralab.data.inventory import sha256_of_file
from respiralab.evaluation.metrics import score_predictions
from respiralab.models.baseline import load_split_features, patient_counts


def load_search_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_svm_pipeline(C: float, gamma: str | float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", random_state=seed),
            ),
        ]
    )


def run_search(
    config: dict, X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray
) -> dict:
    seed = config["seed"]
    c_values = config["param_grid"]["C"]
    gamma_values = config["param_grid"]["gamma"]
    trials = []
    for c_value, gamma_value in product(c_values, gamma_values):
        pipeline = build_svm_pipeline(C=c_value, gamma=gamma_value, seed=seed)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_dev)
        scores = score_predictions(y_dev, y_pred)
        trials.append({"C": c_value, "gamma": gamma_value, **scores})
    best = max(trials, key=lambda t: t["macro_f1"])
    return {"n_trials": len(trials), "trials": trials, "best": {"strategy": "svm_rbf", **best}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D17: bounded RBF SVM hyperparameter search (E02), selected on val only."
    )
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--search-config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest_df = pd.read_csv(args.feature_manifest)
    X_train, y_train = load_split_features(manifest_df, "train", args.cache_dir)
    X_dev, y_dev = load_split_features(manifest_df, "val", args.cache_dir)
    search_config = load_search_config(args.search_config)
    search_result = run_search(search_config, X_train, y_train, X_dev, y_dev)
    result = {
        "split_hash": sha256_of_file(args.splits_csv),
        "patient_counts": patient_counts(manifest_df),
        "n_cycles": {"train": len(y_train), "dev": len(y_dev)},
        "search_config": search_config,
        **search_result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "dev_scores.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(f"ran {search_result['n_trials']} trials (selected on val macro_f1 only)")
    best = search_result["best"]
    print(f"best: C={best['C']} gamma={best['gamma']} macro_f1={best['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

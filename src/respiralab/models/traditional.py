"""D16: train-only StandardScaler + logistic regression pipeline (E01).

sklearn's Pipeline guarantees the scaler is fit only on the data passed to
`.fit()` (train) — scoring on dev later only calls `.transform()`, never
`.fit()`, so the scaler can't see val/test features by construction
(acceptance: "no scaler fits on validation/test"). class_weight="balanced"
on the classifier is docs/04_EXPERIMENTS.md's documented train-only
imbalance-handling convention, needed here since the 4 classes are far
from evenly sized (see D15's E00 support counts).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from respiralab.data.inventory import sha256_of_file
from respiralab.models.baseline import load_split_features, patient_counts, score_predictions

LOGREG_SEED = 42


def build_pipeline(seed: int = LOGREG_SEED) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
            ),
        ]
    )


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_dev)
    return {"strategy": "logistic_regression", **score_predictions(y_dev, y_pred)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D16: train-only scaler + logistic regression (E01)."
    )
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=LOGREG_SEED)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    X_train, y_train = load_split_features(manifest_df, "train", args.cache_dir)
    X_dev, y_dev = load_split_features(manifest_df, "val", args.cache_dir)

    baseline_result = evaluate_pipeline(build_pipeline(args.seed), X_train, y_train, X_dev, y_dev)

    result = {
        "split_hash": sha256_of_file(args.splits_csv),
        "patient_counts": patient_counts(manifest_df),
        "n_cycles": {"train": len(y_train), "dev": len(y_dev)},
        "seed": args.seed,
        "baselines": [baseline_result],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "dev_scores.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(f"logistic_regression: macro_f1={baseline_result['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""D15: majority and stratified dummy baselines (E00).

Reuses sklearn's DummyClassifier for both strategies rather than hand-
rolling majority/stratified prediction — it ignores the feature vectors
entirely and predicts from `label4` alone, which is exactly what a dummy
baseline is. Evaluated on the primary split's val set (used as "dev" per
common ML terminology). Reports sklearn's macro/per-class F1 only —
docs/03_DATA_AND_EVALUATION_CONTRACT.md's primary metric is macro-F1; the
ICBHI-Score/bootstrap metrics module is separate scope (tasks.json D18/D19).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from respiralab.data.inventory import sha256_of_file

LABEL_ORDER = ("normal", "crackle", "wheeze", "both")
DUMMY_SEED = 42
BASELINES = (("majority", "most_frequent"), ("stratified", "stratified"))


def load_split_features(
    manifest_df: pd.DataFrame, split: str, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Stack cached D13 feature vectors + label4 for one split."""
    rows = manifest_df[manifest_df["split"] == split]
    if rows.empty:
        raise ValueError(f"no feature_manifest rows for split={split!r}")
    cache_dir = Path(cache_dir)
    X = np.stack([np.load(cache_dir / p) for p in rows["cache_path"]])
    y = rows["label4"].to_numpy()
    return X, y


def score_predictions(y_dev: np.ndarray, y_pred: np.ndarray) -> dict:
    """macro-F1 (docs/03's primary metric) + per-class precision/recall/F1 + confusion matrix.

    Shared by every model's dev-score report (D15 dummy baselines, D16+
    real classifiers) so the result shape stays identical across E00/E01/...
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_dev, y_pred, labels=LABEL_ORDER, zero_division=0
    )
    cm = confusion_matrix(y_dev, y_pred, labels=LABEL_ORDER)
    return {
        "macro_f1": float(
            f1_score(y_dev, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        ),
        "per_class": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(LABEL_ORDER, precision, recall, f1, support, strict=True)
        },
        "confusion_matrix": {"labels": list(LABEL_ORDER), "matrix": cm.tolist()},
    }


def evaluate_dummy(
    name: str,
    strategy: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
) -> dict:
    clf = DummyClassifier(strategy=strategy, random_state=DUMMY_SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    return {"strategy": name, **score_predictions(y_dev, y_pred)}


def patient_counts(manifest_df: pd.DataFrame) -> dict[str, int]:
    return {
        split: int(manifest_df.loc[manifest_df["split"] == split, "patient_id"].nunique())
        for split in ("train", "val", "test")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D15: majority/stratified dummy baselines (E00).")
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    X_train, y_train = load_split_features(manifest_df, "train", args.cache_dir)
    X_dev, y_dev = load_split_features(manifest_df, "val", args.cache_dir)

    result = {
        "split_hash": sha256_of_file(args.splits_csv),
        "patient_counts": patient_counts(manifest_df),
        "n_cycles": {"train": len(y_train), "dev": len(y_dev)},
        "baselines": [
            evaluate_dummy(name, strategy, X_train, y_train, X_dev, y_dev)
            for name, strategy in BASELINES
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "dev_scores.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    for b in result["baselines"]:
        print(f"{b['strategy']}: macro_f1={b['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

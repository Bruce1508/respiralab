"""D45: temperature scaling + ECE for E01, fit on held-out CV-fold predictions only.

Per ADR-0001 D5, temperature is fit on out-of-fold predictions produced by
`evaluation.group_cv` (each sample scored by a model that never saw it in
training) -- never on the primary test. See
docs/adr/0001-v2-pretrained-encoder-and-evaluation.md.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.pipeline import Pipeline

from respiralab.data.inventory import sha256_of_file
from respiralab.evaluation.group_cv import load_trainval_pool, make_folds
from respiralab.evaluation.metrics import LABEL_ORDER
from respiralab.models.traditional import build_pipeline

CV_SEED = 42
N_SPLITS = 5
N_BINS = 10


def collect_oof_logits(
    X: np.ndarray,
    y: np.ndarray,
    build_fn: Callable[[], Pipeline],
    folds: list[tuple[np.ndarray, np.ndarray]],
    labels: tuple[str, ...] = LABEL_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold logits and true labels: each sample scored by a model fit without it."""
    n = len(y)
    logits = np.full((n, len(labels)), np.nan)
    for train_idx, val_idx in folds:
        pipeline = build_fn()
        pipeline.fit(X[train_idx], y[train_idx])
        fold_logits = pipeline.decision_function(X[val_idx])
        logits[val_idx] = fold_logits[:, [list(pipeline.classes_).index(c) for c in labels]]
    assert not np.isnan(logits).any(), "every sample must be in exactly one fold's held-out set"
    return logits, y


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(temperature: float, logits: np.ndarray, y_idx: np.ndarray) -> float:
    probs = softmax(logits / temperature)
    return float(-np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)).mean())


def fit_temperature(
    logits: np.ndarray, y_true: np.ndarray, labels: tuple[str, ...] = LABEL_ORDER
) -> float:
    """Scalar temperature T>0 minimizing NLL of softmax(logits/T) on held-out data."""
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_idx = np.array([label_to_idx[label] for label in y_true])
    result = minimize_scalar(_nll, args=(logits, y_idx), bounds=(0.05, 20.0), method="bounded")
    return float(result.x)


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, labels: tuple[str, ...] = LABEL_ORDER, n_bins: int = N_BINS
) -> dict:
    """ECE over max-probability confidence, equal-width bins; also returns reliability bins."""
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_idx = np.array([label_to_idx[label] for label in y_true])
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_idx).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (
            confidences <= hi
        )
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0, "accuracy": None, "confidence": None})
            continue
        bin_acc = float(correct[in_bin].mean())
        bin_conf = float(confidences[in_bin].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
        bins.append({"lo": float(lo), "hi": float(hi), "count": count, "accuracy": bin_acc, "confidence": bin_conf})
    return {"ece": float(ece), "n_bins": n_bins, "bins": bins}


def plot_reliability_diagram(bins_before: list[dict], bins_after: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, bins, title in ((axes[0], bins_before, "before (T=1)"), (axes[1], bins_after, "after")):
        confs = [b["confidence"] for b in bins if b["confidence"] is not None]
        accs = [b["accuracy"] for b in bins if b["accuracy"] is not None]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ax.bar(confs, accs, width=0.08, alpha=0.7, edgecolor="black")
        ax.set_xlabel("confidence (max predicted probability)")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D45: temperature scaling + ECE for E01, fit on held-out CV folds."
    )
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--figure-out", type=Path, default=Path("reports/figures/calibration_E01.png"))
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=CV_SEED)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    X, y, patient_ids = load_trainval_pool(manifest_df, args.cache_dir)
    folds = make_folds(patient_ids, y, n_splits=args.n_splits, seed=args.seed)

    logits, y_true = collect_oof_logits(X, y, build_pipeline, folds)
    temperature = fit_temperature(logits, y_true)

    probs_before = softmax(logits)
    probs_after = softmax(logits / temperature)
    ece_before = expected_calibration_error(probs_before, y_true)
    ece_after = expected_calibration_error(probs_after, y_true)

    plot_reliability_diagram(ece_before["bins"], ece_after["bins"], args.figure_out)

    result = {
        "protocol": "temperature scaled on out-of-fold CV predictions (ADR-0001 D5); never on test",
        "split_hash": sha256_of_file(args.splits_csv),
        "n_trainval_patients": len(set(patient_ids)),
        "n_trainval_cycles": len(y),
        "seed": args.seed,
        "model": "logistic_regression (E01 pipeline)",
        "temperature": temperature,
        "ece_before": ece_before["ece"],
        "ece_after": ece_after["ece"],
        "n_bins": N_BINS,
        "bins_before": ece_before["bins"],
        "bins_after": ece_after["bins"],
        "figure": str(args.figure_out),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "calibration_scores.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(f"temperature={temperature:.4f}  ECE before={ece_before['ece']:.4f}  ECE after={ece_after['ece']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

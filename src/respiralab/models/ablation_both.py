"""D50: E07a/b/c single-factor 'both'-class ablations (ADR-0001 D4b).

Reuses D48's cached frozen AST embeddings as-is (no new feature extraction)
and trains a tiny linear head (nn.Linear(768, 4)) under D44's grouped CV.
Each ablation changes exactly one factor relative to the "reference"
(class-weighted CE -- the same technique E01/E05 already use via
sklearn's class_weight='balanced'):

- balanced_sampling: oversample minority classes (WeightedRandomSampler),
  plain (unweighted) CE loss.
- focal_loss: focal loss (gamma=2), no resampling, no class weight.
- mixup: train-only mixup between two random train-fold cycles (interpolated
  embeddings + soft labels), plain CE, no class weight. Both mixup parents
  must be train-fold patients -- checked by an explicit assertion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler

from respiralab.data.inventory import sha256_of_file
from respiralab.evaluation.group_cv import assert_no_patient_crosses_folds, make_folds
from respiralab.evaluation.metrics import LABEL_ORDER, score_predictions
from respiralab.models.encoder import EMBEDDING_DIM
from respiralab.models.encoder_probe import build_trainval_rows
from respiralab.models.training import should_stop_early

LABEL_TO_INDEX = {label: i for i, label in enumerate(LABEL_ORDER)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}
CV_SEED = 42
N_SPLITS = 5
METHODS = ("reference", "balanced_sampling", "focal_loss", "mixup")


def load_cached_embeddings(rows: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    missing = [c for c in rows["cycle_id"] if not (cache_dir / f"{c}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} cycle(s) missing from embedding cache {cache_dir} "
            f"(run D48's encoder_probe extraction first), e.g. {missing[:3]}"
        )
    return np.stack([np.load(cache_dir / f"{c}.npy") for c in rows["cycle_id"]]).astype(np.float32)


def class_weight_tensor(y_idx: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -((1 - pt) ** gamma * log_pt).mean()


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def mixup_pair(
    X: torch.Tensor,
    y_onehot: torch.Tensor,
    patient_ids: np.ndarray,
    train_patients: set,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolates each sample with a random other sample from the same batch.

    Contract: every patient mixed in must already be a train-fold patient --
    asserted explicitly so a caller can never accidentally mix in val data.
    """
    if not set(patient_ids).issubset(train_patients):
        raise AssertionError("mixup parent(s) outside the train-fold patient set")
    n = X.shape[0]
    perm = torch.randperm(n)
    lam = float(np.random.beta(alpha, alpha))
    X_mixed = lam * X + (1 - lam) * X[perm]
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[perm]
    return X_mixed, y_mixed


def make_sampler(y_idx: np.ndarray, n_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    class_weight = 1.0 / np.maximum(counts, 1)
    sample_weight = class_weight[y_idx]
    return WeightedRandomSampler(
        torch.tensor(sample_weight, dtype=torch.double), num_samples=len(y_idx), replacement=True
    )


def train_probe(
    X_train: np.ndarray,
    y_train_idx: np.ndarray,
    patient_ids_train: np.ndarray,
    X_val: np.ndarray,
    y_val_idx: np.ndarray,
    method: str,
    max_epochs: int = 50,
    patience: int = 5,
    lr: float = 1e-2,
    batch_size: int = 64,
    seed: int = CV_SEED,
) -> dict:
    """Trains nn.Linear(EMBEDDING_DIM, n_classes) with `method`'s single factor, early-stopped on val macro-F1."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, must be one of {METHODS}")
    torch.manual_seed(seed)
    n_classes = len(LABEL_ORDER)
    model = nn.Linear(EMBEDDING_DIM, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_idx, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    class_weights = class_weight_tensor(y_train_idx, n_classes)
    train_patient_set = set(patient_ids_train)
    sampler = make_sampler(y_train_idx, n_classes) if method == "balanced_sampling" else None

    n = len(y_train_t)
    best_val_macro_f1 = -1.0
    best_scores: dict | None = None
    epochs_without_improvement = 0
    history: list[dict] = []
    for epoch in range(max_epochs):
        model.train()
        order = list(sampler) if sampler is not None else torch.randperm(n).tolist()
        for start in range(0, n, batch_size):
            batch_idx = torch.tensor(order[start : start + batch_size])
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            optimizer.zero_grad()
            if method == "mixup":
                y_onehot = F.one_hot(y_batch, n_classes).float()
                X_batch, y_onehot = mixup_pair(
                    X_batch, y_onehot, patient_ids_train[batch_idx.numpy()], train_patient_set
                )
                loss = soft_cross_entropy(model(X_batch), y_onehot)
            elif method == "focal_loss":
                loss = focal_loss(model(X_batch), y_batch)
            elif method == "balanced_sampling":
                loss = F.cross_entropy(model(X_batch), y_batch)
            else:  # reference: class-weighted, matches E01/E05's class_weight='balanced'
                loss = F.cross_entropy(model(X_batch), y_batch, weight=class_weights)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred_idx = model(X_val_t).argmax(dim=1).numpy()
        val_pred = np.array([INDEX_TO_LABEL[i] for i in val_pred_idx])
        val_true = np.array([INDEX_TO_LABEL[i] for i in y_val_idx])
        val_scores = score_predictions(val_true, val_pred)
        best_val_macro_f1, epochs_without_improvement, stop = should_stop_early(
            best_val_macro_f1, val_scores["macro_f1"], epochs_without_improvement, patience
        )
        if epochs_without_improvement == 0:
            best_scores = val_scores
        history.append({"epoch": epoch, "val_macro_f1": val_scores["macro_f1"]})
        if stop:
            break

    assert best_scores is not None
    return {
        "n_epochs_run": len(history),
        "best_val_macro_f1": best_scores["macro_f1"],
        "per_class": best_scores["per_class"],
    }


def run_cv(
    X: np.ndarray, y_idx: np.ndarray, patient_ids: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], method: str
) -> dict:
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        run = train_probe(
            X[train_idx],
            y_idx[train_idx],
            patient_ids[train_idx],
            X[val_idx],
            y_idx[val_idx],
            method=method,
        )
        fold_results.append({"fold": fold_idx, **run})
    macro_f1s = np.array([f["best_val_macro_f1"] for f in fold_results])
    both_recalls = np.array([f["per_class"]["both"]["recall"] for f in fold_results])
    return {
        "method": method,
        "folds": fold_results,
        "macro_f1_mean": float(macro_f1s.mean()),
        "macro_f1_sd": float(macro_f1s.std(ddof=1)),
        "both_recall_mean": float(both_recalls.mean()),
        "both_recall_sd": float(both_recalls.std(ddof=1)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D50: E07a/b/c single-factor 'both'-class ablations vs class-weighted reference."
    )
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cycles-csv", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    cycles_df = pd.read_csv(args.cycles_csv)
    rows = build_trainval_rows(manifest_df, cycles_df)
    X = load_cached_embeddings(rows, args.embedding_cache)
    y_idx = np.array([LABEL_TO_INDEX[label] for label in rows["label4"]])
    patient_ids = rows["patient_id"].astype(str).to_numpy()

    folds = make_folds(patient_ids, y_idx, n_splits=args.n_splits, seed=CV_SEED)
    assert_no_patient_crosses_folds(patient_ids, folds)

    methods_result = {method: run_cv(X, y_idx, patient_ids, folds, method) for method in METHODS}

    result = {
        "protocol": "D50: E07a/b/c single-factor 'both'-class ablations vs class-weighted reference (ADR-0001 D4b)",
        "split_hash": sha256_of_file(args.splits_csv),
        "fold_split_seed": CV_SEED,
        "n_trainval_cycles": len(y_idx),
        "methods": methods_result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "ablation_scores.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    for method in METHODS:
        r = methods_result[method]
        print(
            f"{method}: macro_f1={r['macro_f1_mean']:.4f}+/-{r['macro_f1_sd']:.4f} "
            f"both_recall={r['both_recall_mean']:.4f}+/-{r['both_recall_sd']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

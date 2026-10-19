"""D52: freeze v2 (E06) -- Gate G7: out-of-fold calibration + final weights hash.

Model choice (E06, bounded fine-tune) traces to results/E06/grid_scores.json
(D49: macro_f1=0.4791+/-0.0401, clears E05's mean+1sd bar) -- decided with the
user, not here. This module only (a) fits a temperature on out-of-fold val
logits (same technique D45 used for E01, never touching the primary test),
and (b) retrains the exact same architecture/hyperparams on 100% of the
train+val pool for a fixed epoch count (the median best-epoch count observed
across D49's 15 CV runs -- see reports/v2_model_freeze.md), then hashes the
resulting weights for provenance. No primary-test numbers are read or cited
anywhere in this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor

from respiralab.data.inventory import sha256_of_file
from respiralab.evaluation.calibration import expected_calibration_error, fit_temperature, softmax
from respiralab.models.encoder import CHECKPOINT, checkpoint_sha256
from respiralab.models.encoder_probe import build_trainval_rows
from respiralab.models.finetune import (
    ASTClassifier,
    RawCycleDataset,
    collate_raw,
    git_commit,
    run_epoch,
    run_fold,
    select_device,
)

FINAL_MODEL_SEED = 42
UNFREEZE_LAST_N = 2
LR = 1e-5
BATCH_SIZE = 8
MAX_EPOCHS_CV = 3  # matches D49's grid hyperparams, for the calibration OOF runs
PATIENCE = 1
FINAL_EPOCHS = 2  # median best-epoch count across D49's 15 grid runs (see reports/v2_model_freeze.md)
N_SPLITS = 5


def calibrate(
    rows: pd.DataFrame, audio_dir: Path, device: torch.device, feature_extractor: ASTFeatureExtractor
) -> dict:
    """Out-of-fold temperature calibration: 5 folds, same model_seed as the final frozen model."""
    cv_args = argparse.Namespace(
        n_splits=N_SPLITS,
        unfreeze_last_n_layers=UNFREEZE_LAST_N,
        max_epochs=MAX_EPOCHS_CV,
        patience=PATIENCE,
        lr=LR,
        batch_size=BATCH_SIZE,
        audio_dir=audio_dir,
    )
    all_logits = []
    all_labels = []
    for fold in range(N_SPLITS):
        run = run_fold(rows, fold, FINAL_MODEL_SEED, cv_args, device, feature_extractor, track_best_state=True)
        all_logits.append(run["val_logits"])
        all_labels.append(run["val_labels"])
        print(f"[calibrate fold={fold}] best_val_macro_f1={run['best_val_macro_f1']:.4f}")
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    temperature = fit_temperature(logits, y_true)
    ece_before = expected_calibration_error(softmax(logits), y_true)
    ece_after = expected_calibration_error(softmax(logits / temperature), y_true)
    return {
        "n_oof_cycles": len(y_true),
        "temperature": temperature,
        "ece_before": ece_before["ece"],
        "ece_after": ece_after["ece"],
    }


def train_final(
    rows: pd.DataFrame, audio_dir: Path, out_dir: Path, device: torch.device, feature_extractor: ASTFeatureExtractor
) -> dict:
    """Retrains on 100% of train+val for FINAL_EPOCHS (fixed, no val split left to early-stop against)."""
    torch.manual_seed(FINAL_MODEL_SEED)
    model = ASTClassifier(unfreeze_last_n=UNFREEZE_LAST_N).to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=LR)
    dataset = RawCycleDataset(rows, audio_dir)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_raw(batch, feature_extractor),
    )

    history = []
    for epoch in range(FINAL_EPOCHS):
        train_scores = run_epoch(model, loader, device, optimizer=optimizer)
        history.append(
            {"epoch": epoch, "train_macro_f1": train_scores["macro_f1"], "train_loss": train_scores["loss"]}
        )
        print(f"[final train epoch={epoch}] train_macro_f1={train_scores['macro_f1']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "v2_model_state_dict.pt"
    torch.save(model.state_dict(), weights_path)
    return {
        "weights_path": str(weights_path),
        "weights_sha256": sha256_of_file(weights_path),
        "n_train_cycles": len(rows),
        "history": history,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D52: freeze v2 (E06) -- calibration + final weights.")
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cycles-csv", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    cycles_df = pd.read_csv(args.cycles_csv)
    rows = build_trainval_rows(manifest_df, cycles_df)

    device = select_device()
    feature_extractor = ASTFeatureExtractor.from_pretrained(CHECKPOINT)

    calibration = calibrate(rows, args.audio_dir, device, feature_extractor)
    final = train_final(rows, args.audio_dir, args.out, device, feature_extractor)

    result = {
        "protocol": "D52: freeze v2 (E06) -- Gate G7. No primary-test numbers used.",
        "git_commit": git_commit(),
        "device": str(device),
        "split_hash": sha256_of_file(args.splits_csv),
        "encoder_checkpoint": CHECKPOINT,
        "encoder_checkpoint_sha256": checkpoint_sha256(),
        "final_model_seed": FINAL_MODEL_SEED,
        "unfreeze_last_n_layers": UNFREEZE_LAST_N,
        "final_epochs": FINAL_EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "calibration": calibration,
        "final_model": final,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "v2_freeze.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(
        f"temperature={calibration['temperature']:.4f} "
        f"ECE {calibration['ece_before']:.4f}->{calibration['ece_after']:.4f} "
        f"weights_sha256={final['weights_sha256'][:16]}..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

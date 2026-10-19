"""D49 pilot: bounded fine-tune of the AST encoder (E06), ADR-0001 D4b/D8.

Pilot scope (1 seed x 1 fold x N epochs, N small): measure a real wall-clock
and macro-F1 signal before committing to the full 3-seed x 5-fold grid. Only
the last `--unfreeze-last-n-layers` AST transformer blocks + the final
layernorm + a new classifier head are trained; the rest of the encoder stays
frozen. Raw audio is re-extracted every epoch (D48's embedding cache is NOT
reused here) because the encoder's weights change.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor, ASTModel

from respiralab.data.inventory import sha256_of_file
from respiralab.dsp.encoder_input import ENCODER_SAMPLE_RATE, prepare_encoder_input
from respiralab.dsp.segment import extract_cycle
from respiralab.evaluation.group_cv import assert_no_patient_crosses_folds, make_folds
from respiralab.evaluation.metrics import LABEL_ORDER, score_predictions
from respiralab.models.encoder import CHECKPOINT, EMBEDDING_DIM
from respiralab.models.encoder_probe import build_trainval_rows
from respiralab.models.training import should_stop_early

LABEL_TO_INDEX = {label: i for i, label in enumerate(LABEL_ORDER)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}
N_SPLITS = 5
CV_SEED = 42  # fold-splitting seed -- fixed, matches D44/D48; must NOT vary with the model seed below
E06_MODEL_SEEDS = (13, 21, 42)  # ADR-0001 D4b pre-registered seeds for E06's robustness check


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ASTClassifier(nn.Module):
    """Pretrained AST encoder with only its last N blocks unfrozen + a new head."""

    def __init__(self, unfreeze_last_n: int = 2, n_classes: int = len(LABEL_ORDER)):
        super().__init__()
        self.encoder = ASTModel.from_pretrained(CHECKPOINT)
        self.encoder.eval()  # dropout off everywhere, incl. the unfrozen blocks -- deterministic forward
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.encoder.layers[-unfreeze_last_n:].parameters():
            p.requires_grad_(True)
        for p in self.encoder.layernorm.parameters():
            p.requires_grad_(True)
        self.classifier = nn.Linear(EMBEDDING_DIM, n_classes)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, inputs: dict) -> torch.Tensor:
        pooled = self.encoder(**inputs).pooler_output
        return self.classifier(pooled)


class RawCycleDataset(Dataset):
    """Raw (resampled/filtered/padded) waveforms for a fold's row subset.

    Unlike D48's embedding cache, this re-extracts raw audio every access --
    fine-tuning needs it fresh each epoch since the encoder weights change.
    """

    def __init__(self, rows: pd.DataFrame, audio_dir: Path, pad_mode: str = "zero"):
        self.audio_dir = audio_dir
        self.pad_mode = pad_mode
        self.rows = rows.reset_index(drop=True)
        self._audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _read_record(self, record_id: str) -> tuple[np.ndarray, int]:
        if record_id not in self._audio_cache:
            if len(self._audio_cache) > 50:  # ponytail: bound memory, drop an old entry
                self._audio_cache.pop(next(iter(self._audio_cache)))
            self._audio_cache[record_id] = sf.read(str(self.audio_dir / f"{record_id}.wav"))
        return self._audio_cache[record_id]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        row = self.rows.iloc[idx]
        audio, sr = self._read_record(row["record_id"])
        cycle_audio = extract_cycle(audio, sr, row["start_s"], row["end_s"])
        prepped = prepare_encoder_input(cycle_audio, sr, pad_mode=self.pad_mode)
        return prepped.astype(np.float32), LABEL_TO_INDEX[row["label4"]]


def collate_raw(batch, feature_extractor: ASTFeatureExtractor) -> tuple[dict, torch.Tensor]:
    waveforms, labels = zip(*batch, strict=True)
    inputs = feature_extractor(list(waveforms), sampling_rate=ENCODER_SAMPLE_RATE, return_tensors="pt")
    return dict(inputs), torch.tensor(labels, dtype=torch.long)


def run_epoch(model: ASTClassifier, loader, device: torch.device, optimizer=None) -> dict:
    """One pass over loader. Trains (grad) if optimizer is given, else evaluates (no_grad)."""
    train = optimizer is not None
    loss_fn = nn.CrossEntropyLoss()
    all_true: list[int] = []
    all_pred: list[int] = []
    total_loss, n = 0.0, 0
    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(inputs)
                loss = loss_fn(logits, labels)
        total_loss += loss.item() * len(labels)
        n += len(labels)
        all_true.extend(labels.tolist())
        all_pred.extend(logits.argmax(dim=1).tolist())
    y_true = np.array([INDEX_TO_LABEL[i] for i in all_true])
    y_pred = np.array([INDEX_TO_LABEL[i] for i in all_pred])
    scores = score_predictions(y_true, y_pred)
    scores["loss"] = total_loss / max(n, 1)
    return scores


def collect_logits(model: ASTClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Raw per-class logits + true labels for every item in loader, in loader order."""
    all_logits: list[np.ndarray] = []
    all_true: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            all_logits.append(model(inputs).cpu().numpy())
            all_true.extend(labels.tolist())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.array([INDEX_TO_LABEL[i] for i in all_true])
    return logits, y_true


def run_fold(
    rows: pd.DataFrame,
    fold: int,
    model_seed: int,
    args: argparse.Namespace,
    device: torch.device,
    feature_extractor: ASTFeatureExtractor,
    track_best_state: bool = False,
) -> dict:
    """Train+validate one (fold, model_seed) run with validation-based early stopping.

    With track_best_state=True, the model is rolled back to its best-val-macro-F1
    epoch's weights before returning, and that epoch's val logits/labels are
    included -- used for D52's out-of-fold calibration, not the CV grid itself.
    """
    y = rows["label4"].to_numpy()
    patient_ids = rows["patient_id"].astype(str).to_numpy()
    folds = make_folds(patient_ids, y, n_splits=args.n_splits, seed=CV_SEED)
    assert_no_patient_crosses_folds(patient_ids, folds)
    train_idx, val_idx = folds[fold]
    train_rows, val_rows = rows.iloc[train_idx], rows.iloc[val_idx]

    torch.manual_seed(model_seed)
    model = ASTClassifier(unfreeze_last_n=args.unfreeze_last_n_layers).to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)

    def make_loader(rows_subset: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = RawCycleDataset(rows_subset, args.audio_dir)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_raw(batch, feature_extractor),
        )

    train_loader = make_loader(train_rows, shuffle=True)
    val_loader = make_loader(val_rows, shuffle=False)

    start = time.monotonic()
    history: list[dict] = []
    best_val_macro_f1 = -1.0
    best_state_dict = None
    epochs_without_improvement = 0
    for epoch in range(args.max_epochs):
        train_scores = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_scores = run_epoch(model, val_loader, device, optimizer=None)
        best_val_macro_f1, epochs_without_improvement, stop = should_stop_early(
            best_val_macro_f1, val_scores["macro_f1"], epochs_without_improvement, args.patience
        )
        if track_best_state and epochs_without_improvement == 0:
            best_state_dict = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_macro_f1": train_scores["macro_f1"],
                "train_loss": train_scores["loss"],
                "val_macro_f1": val_scores["macro_f1"],
                "val_loss": val_scores["loss"],
            }
        )
        if stop:
            break
    runtime_s = time.monotonic() - start

    result = {
        "fold": fold,
        "model_seed": model_seed,
        "n_train_cycles": len(train_idx),
        "n_val_cycles": len(val_idx),
        "n_epochs_run": len(history),
        "runtime_s": runtime_s,
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
    }
    if track_best_state:
        assert best_state_dict is not None
        model.load_state_dict(best_state_dict)
        val_logits, val_labels = collect_logits(model, val_loader, device)
        result["val_logits"] = val_logits
        result["val_labels"] = val_labels
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D49: E06 bounded fine-tune of the AST encoder (pilot run or the full grid)."
    )
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cycles-csv", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--e01-cv-scores", required=True, type=Path)
    parser.add_argument("--e05-cv-scores", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--grid", action="store_true", help="run all 3 seeds x n-splits folds")
    parser.add_argument("--fold", type=int, default=0, help="single-run mode only (ignored with --grid)")
    parser.add_argument(
        "--model-seed", type=int, default=42, help="single-run mode only (ignored with --grid)"
    )
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    cycles_df = pd.read_csv(args.cycles_csv)
    rows = build_trainval_rows(manifest_df, cycles_df)

    device = select_device()
    feature_extractor = ASTFeatureExtractor.from_pretrained(CHECKPOINT)
    e01_cv = json.loads(args.e01_cv_scores.read_text())
    e05_cv = json.loads(args.e05_cv_scores.read_text())

    plan = (
        [(seed, fold) for seed in E06_MODEL_SEEDS for fold in range(args.n_splits)]
        if args.grid
        else [(args.model_seed, args.fold)]
    )

    runs = []
    for model_seed, fold in plan:
        run = run_fold(rows, fold, model_seed, args, device, feature_extractor)
        e01_fold = next(f for f in e01_cv["folds"] if f["fold"] == fold)
        e05_fold = next(f for f in e05_cv["folds"] if f["fold"] == fold)
        run["same_fold_e01_macro_f1"] = e01_fold["macro_f1"]
        run["same_fold_e05_macro_f1"] = e05_fold["macro_f1"]
        runs.append(run)
        print(
            f"[seed={model_seed} fold={fold}] best val macro_f1={run['best_val_macro_f1']:.4f} "
            f"(E01={e01_fold['macro_f1']:.4f}, E05={e05_fold['macro_f1']:.4f}) "
            f"in {run['runtime_s']:.1f}s, {run['n_epochs_run']} epoch(s)"
        )

    best_f1s = np.array([r["best_val_macro_f1"] for r in runs])
    result = {
        "protocol": (
            f"D49 {'grid' if args.grid else 'single run'}: E06 bounded fine-tune "
            f"(ADR-0001 D4b/D8), {len(runs)} run(s)"
        ),
        "checkpoint": CHECKPOINT,
        "git_commit": git_commit(),
        "device": str(device),
        "cpu_count": os.cpu_count(),
        "split_hash": sha256_of_file(args.splits_csv),
        "fold_split_seed": CV_SEED,
        "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "runs": runs,
        "macro_f1_mean": float(best_f1s.mean()),
        "macro_f1_sd": float(best_f1s.std(ddof=1)) if len(best_f1s) > 1 else 0.0,
        "e01_cv_macro_f1_mean": e01_cv["macro_f1_mean"],
        "e05_cv_macro_f1_mean": e05_cv["macro_f1_mean"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_name = "grid_scores.json" if args.grid else "pilot_scores.json"
    out_path = args.out / out_name
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(
        f"E06 macro_f1={result['macro_f1_mean']:.4f}+/-{result['macro_f1_sd']:.4f} "
        f"vs E01 CV mean={result['e01_cv_macro_f1_mean']:.4f}, "
        f"E05 CV mean={result['e05_cv_macro_f1_mean']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

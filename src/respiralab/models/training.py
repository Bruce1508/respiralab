from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from respiralab.data.inventory import sha256_of_file
from respiralab.evaluation.metrics import LABEL_ORDER, score_predictions
from respiralab.models.cnn import CompactCNN, collate_batch
from respiralab.models.dataset import LogMelCycleDataset

INDEX_TO_LABEL = {i: label for i, label in enumerate(LABEL_ORDER)}


def evaluate(model: nn.Module, loader) -> dict:
    model.eval()
    all_true: list[str] = []
    all_pred: list[str] = []
    total_loss, n = 0.0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["log_mel"], batch["mask"])
            loss = loss_fn(logits, batch["label"])
            total_loss += loss.item() * len(batch["label"])
            n += len(batch["label"])
            preds = logits.argmax(dim=1)
            all_true.extend(INDEX_TO_LABEL[i] for i in batch["label"].tolist())
            all_pred.extend(INDEX_TO_LABEL[i] for i in preds.tolist())
    scores = score_predictions(np.array(all_true), np.array(all_pred))
    scores["loss"] = total_loss / max(n, 1)
    return scores


def train_one_epoch(model: nn.Module, loader, optimizer, loss_fn) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch["log_mel"], batch["mask"])
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch["label"])
        n += len(batch["label"])
    return total_loss / max(n, 1)


def should_stop_early(
    best_val_metric: float, current_val_metric: float, epochs_without_improvement: int, patience: int
) -> tuple[float, int, bool]:
    if current_val_metric > best_val_metric:
        return current_val_metric, 0, False
    epochs_without_improvement += 1
    return best_val_metric, epochs_without_improvement, epochs_without_improvement >= patience


def save_checkpoint(path: Path, **state) -> None:
    torch.save(state, path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, weights_only=True)


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    max_epochs: int,
    lr: float,
    patience: int,
    checkpoint_path: Path | None = None,
    resume_state: dict | None = None,
    evaluate_fn: Callable[[nn.Module, object], dict] = evaluate,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict] = []
    start_epoch = 0
    best_val_macro_f1 = -1.0
    best_state_dict = None
    epochs_without_improvement = 0

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        start_epoch = resume_state["epoch"]
        best_val_macro_f1 = resume_state["best_val_macro_f1"]
        best_state_dict = resume_state["best_state_dict"]
        epochs_without_improvement = resume_state["epochs_without_improvement"]
        history = resume_state["history"]

    stopped_early = False
    for epoch in range(start_epoch, max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_scores = evaluate_fn(model, val_loader)
        val_macro_f1 = val_scores["macro_f1"]

        best_val_macro_f1, epochs_without_improvement, stopped_early = should_stop_early(
            best_val_macro_f1, val_macro_f1, epochs_without_improvement, patience
        )
        improved = epochs_without_improvement == 0
        if improved:
            best_state_dict = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_scores["loss"],
                "val_macro_f1": val_macro_f1,
                "improved": improved,
            }
        )

        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch + 1,
                best_val_macro_f1=best_val_macro_f1,
                best_state_dict=best_state_dict,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
            )

        if stopped_early:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "best_state_dict": best_state_dict if best_state_dict is not None else model.state_dict(),
        "stopped_early": stopped_early,
    }


def build_split_loader(
    cycles_csv: Path, splits_csv: Path, audio_dir: Path, split: str, batch_size: int
) -> DataLoader:
    cycles_df = pd.read_csv(cycles_csv)
    splits_df = pd.read_csv(splits_csv)
    merged = cycles_df.merge(splits_df[["record_id", "split"]], on="record_id")
    split_rows = merged[merged["split"] == split].reset_index(drop=True)
    dataset = LogMelCycleDataset(split_rows, audio_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate_batch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D24: CNN training loop with validation-only early stopping and resume."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cycles-csv", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    torch.manual_seed(config["seed"])

    train_loader = build_split_loader(
        args.cycles_csv, args.splits_csv, args.audio_dir, "train", config["batch_size"]
    )
    val_loader = build_split_loader(
        args.cycles_csv, args.splits_csv, args.audio_dir, "val", config["batch_size"]
    )

    model = CompactCNN()
    resume_state = load_checkpoint(args.resume_from) if args.resume_from else None

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "checkpoint.pt"
    result = fit(
        model,
        train_loader,
        val_loader,
        max_epochs=config["max_epochs"],
        lr=config["lr"],
        patience=config["patience"],
        checkpoint_path=checkpoint_path,
        resume_state=resume_state,
    )

    summary = {
        "split_hash": sha256_of_file(args.splits_csv),
        "config": config,
        "resumed_from": str(args.resume_from) if args.resume_from else None,
        "n_epochs_run": len(result["history"]),
        "stopped_early": result["stopped_early"],
        "best_val_macro_f1": result["best_val_macro_f1"],
        "history": result["history"],
        "checkpoint_path": str(checkpoint_path),
    }
    out_path = args.out / "training_log.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    print(
        f"ran {summary['n_epochs_run']} epochs (stopped_early={summary['stopped_early']}), "
        f"best val macro_f1={summary['best_val_macro_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

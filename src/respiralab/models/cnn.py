from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from respiralab.data.inventory import sha256_of_file
from respiralab.evaluation.metrics import LABEL_ORDER
from respiralab.models.dataset import LogMelCycleDataset

N_CLASSES = len(LABEL_ORDER)
SMOKE_N_PER_CLASS = 4
SMOKE_N_STEPS = 100
SMOKE_LR = 1e-2


class CompactCNN(nn.Module):
    """Small 2D CNN over fixed-shape log-Mel cycles.

    Convolutions only pool over the mel axis (stride 1 in time), so the
    original per-frame mask from LogMelCycleDataset still applies at the
    final pooling step without needing to be downsampled.
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, log_mel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # zero padded frames before the conv so its 3x3 kernel can't leak
        # whatever garbage sits past the real-audio boundary into nearby frames
        log_mel = log_mel.masked_fill(~mask.unsqueeze(1), 0.0)
        x = self.conv(log_mel.unsqueeze(1))  # (B, C, n_mel', T)
        x = x.mean(dim=2)  # average over remaining mel bins -> (B, C, T)
        mask_f = mask.float().unsqueeze(1)  # (B, 1, T)
        n_real = mask_f.sum(dim=2).clamp(min=1.0)
        pooled = (x * mask_f).sum(dim=2) / n_real  # masked average over time -> (B, C)
        return self.classifier(pooled)


def train_smoke_batch(
    model: nn.Module, batch: dict, n_steps: int, lr: float, seed: int
) -> list[float]:
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        logits = model(batch["log_mel"], batch["mask"])
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def load_smoke_cycles(
    cycles_csv: Path, splits_csv: Path, split: str, n_per_class: int, seed: int
) -> pd.DataFrame:
    cycles_df = pd.read_csv(cycles_csv)
    splits_df = pd.read_csv(splits_csv)
    merged = cycles_df.merge(splits_df[["record_id", "split"]], on="record_id")
    split_rows = merged[merged["split"] == split]
    rng = np.random.RandomState(seed)
    picked = [
        group.sample(n=min(n_per_class, len(group)), random_state=rng)
        for _, group in split_rows.groupby("label4")
    ]
    return pd.concat(picked, ignore_index=True)


def collate_batch(items: list[dict]) -> dict:
    return {
        "log_mel": torch.stack([item["log_mel"] for item in items]),
        "mask": torch.stack([item["mask"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D23: compact CNN overfit smoke test on a tiny real-data batch (E03)."
    )
    parser.add_argument("--cycles-csv", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cycles_df = load_smoke_cycles(
        args.cycles_csv, args.splits_csv, "train", SMOKE_N_PER_CLASS, args.seed
    )
    dataset = LogMelCycleDataset(cycles_df, args.audio_dir)
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_batch
    )
    batch = next(iter(loader))

    model = CompactCNN(n_classes=N_CLASSES)
    losses = train_smoke_batch(model, batch, n_steps=SMOKE_N_STEPS, lr=SMOKE_LR, seed=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "smoke_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)

    result = {
        "split_hash": sha256_of_file(args.splits_csv),
        "n_cycles": len(cycles_df),
        "label_counts": cycles_df["label4"].value_counts().to_dict(),
        "config": {"n_steps": SMOKE_N_STEPS, "lr": SMOKE_LR, "seed": args.seed},
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "checkpoint_path": str(checkpoint_path),
    }
    out_path = args.out / "smoke_run.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} over {SMOKE_N_STEPS} steps on {len(cycles_df)} cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

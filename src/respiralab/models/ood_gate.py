"""D56: train-fit OOD/input-quality gate for v2 (ADR-0001 D7).

Mahalanobis distance of a cycle's pooled AST embedding -- from v2's own
fine-tuned encoder, not E05's frozen one, since the embedding space itself
changed once the last 2 blocks were unfrozen at D49 -- to the train+val
pool's embedding distribution. Threshold fit on train only, never on test
or external data. `inference.v2_predict` uses this to refuse a prediction
on off-distribution input instead of silently extrapolating.
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
from respiralab.models.encoder import CHECKPOINT
from respiralab.models.encoder_probe import build_trainval_rows
from respiralab.models.finetune import ASTClassifier, RawCycleDataset, collate_raw, select_device

UNFREEZE_LAST_N = 2  # must match D52's frozen architecture
BATCH_SIZE = 8
THRESHOLD_PERCENTILE = 99.0  # reject a cycle whose distance exceeds this percentile of train distances
COV_EPSILON = 1e-6  # ponytail: diagonal shrinkage for numerical stability, not full Ledoit-Wolf


def pooled_embeddings(model: ASTClassifier, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Pre-classifier pooled encoder output for every item in loader, in loader order."""
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out.append(model.encoder(**inputs).pooler_output.cpu().numpy())
    return np.concatenate(out, axis=0)


def mahalanobis_distances(embeddings: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diffs = embeddings - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs))


def fit_gate(embeddings: np.ndarray, percentile: float = THRESHOLD_PERCENTILE) -> dict:
    """Mean + regularized inverse covariance + a train-distance threshold. Train data only."""
    mean = embeddings.mean(axis=0)
    cov = np.cov(embeddings, rowvar=False) + COV_EPSILON * np.eye(embeddings.shape[1])
    cov_inv = np.linalg.inv(cov)
    distances = mahalanobis_distances(embeddings, mean, cov_inv)
    return {
        "mean": mean,
        "cov_inv": cov_inv,
        "threshold": float(np.percentile(distances, percentile)),
        "percentile": percentile,
        "n_train_cycles": len(embeddings),
        "train_distance_mean": float(distances.mean()),
        "train_distance_max": float(distances.max()),
    }


def is_in_distribution(embedding: np.ndarray, gate: dict) -> tuple[bool, float]:
    distance = float(
        mahalanobis_distances(embedding.reshape(1, -1), np.asarray(gate["mean"]), np.asarray(gate["cov_inv"]))[0]
    )
    return distance <= gate["threshold"], distance


def save_gate(gate: dict, out_dir: Path) -> dict:
    """Mean/cov_inv as .npy artifacts (not embedded JSON -- cov_inv alone is 768x768 floats),
    metadata (threshold, provenance hashes) as a small JSON, same pattern as v2_freeze.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_path = out_dir / "ood_gate_mean.npy"
    cov_inv_path = out_dir / "ood_gate_cov_inv.npy"
    np.save(mean_path, gate["mean"])
    np.save(cov_inv_path, gate["cov_inv"])
    return {
        "mean_path": str(mean_path),
        "mean_sha256": sha256_of_file(mean_path),
        "cov_inv_path": str(cov_inv_path),
        "cov_inv_sha256": sha256_of_file(cov_inv_path),
        "threshold": gate["threshold"],
        "percentile": gate["percentile"],
        "n_train_cycles": gate["n_train_cycles"],
        "train_distance_mean": gate["train_distance_mean"],
        "train_distance_max": gate["train_distance_max"],
    }


def load_gate(gate_metadata: dict) -> dict:
    """Loads a gate dict with numpy mean/cov_inv back from the metadata JSON's artifact paths."""
    return {
        "mean": np.load(gate_metadata["mean_path"]),
        "cov_inv": np.load(gate_metadata["cov_inv_path"]),
        "threshold": gate_metadata["threshold"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D56: fit the train-only OOD gate for frozen v2 (E06).")
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cycles-csv", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--v2-freeze-json", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    cycles_df = pd.read_csv(args.cycles_csv)
    rows = build_trainval_rows(manifest_df, cycles_df)

    freeze = json.loads(args.v2_freeze_json.read_text())
    device = select_device()
    feature_extractor = ASTFeatureExtractor.from_pretrained(CHECKPOINT)
    model = ASTClassifier(unfreeze_last_n=UNFREEZE_LAST_N).to(device)
    model.load_state_dict(
        torch.load(freeze["final_model"]["weights_path"], map_location=device, weights_only=True)
    )

    dataset = RawCycleDataset(rows, args.audio_dir)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_raw(batch, feature_extractor),
    )
    embeddings = pooled_embeddings(model, loader, device)

    gate = fit_gate(embeddings)
    metadata = save_gate(gate, args.out.parent)
    metadata["weights_sha256_used"] = freeze["final_model"]["weights_sha256"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metadata, indent=2))
    print(
        f"wrote {args.out}: threshold={metadata['threshold']:.3f} (p{gate['percentile']}), "
        f"n_train={metadata['n_train_cycles']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

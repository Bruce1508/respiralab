"""D48: E05 -- frozen encoder embeddings -> logistic regression, under D44's grouped CV.

Per ADR-0001 D4b: compare this probe's CV-mean macro-F1 against E01's grouped-CV
reference (results/E01_cv/cv_scores.json). Go/stop rule for E06 (fine-tuning):
proceed only if E05 clears E01's CV mean + 1 fold-sd.

Embeddings are cached per cycle_id (gitignored, resumable) since each forward
pass through the AST encoder costs real wall-clock time on CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from respiralab.data.inventory import sha256_of_file
from respiralab.dsp.encoder_input import prepare_encoder_input
from respiralab.dsp.segment import extract_cycle
from respiralab.evaluation.group_cv import (
    assert_no_patient_crosses_folds,
    make_folds,
    run_grouped_cv,
)
from respiralab.models.encoder import CHECKPOINT, EMBEDDING_DIM, embed_batch
from respiralab.models.traditional import build_pipeline

CV_SEED = 42
N_SPLITS = 5
BATCH_SIZE = 32


def build_trainval_rows(manifest_df: pd.DataFrame, cycles_df: pd.DataFrame) -> pd.DataFrame:
    """Same row set/order as group_cv.load_trainval_pool, enriched with cycle bounds."""
    rows = manifest_df[manifest_df["split"].isin(("train", "val"))].copy()
    if rows.empty:
        raise ValueError("no feature_manifest rows for split in {train, val}")
    bounds = cycles_df.set_index("cycle_id")[["record_id", "start_s", "end_s"]]
    rows = rows.join(bounds, on="cycle_id", rsuffix="_bounds")
    if rows["start_s"].isna().any():
        missing = rows.loc[rows["start_s"].isna(), "cycle_id"].tolist()
        raise ValueError(f"cycle_id(s) missing from cycles.csv: {missing[:5]}")
    return rows


def extract_and_cache_embeddings(
    rows: pd.DataFrame, audio_dir: Path, cache_dir: Path, pad_mode: str = "zero"
) -> np.ndarray:
    """Per-cycle cached embeddings, in rows' order. Resumable: skips already-cached cycles."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.empty((len(rows), EMBEDDING_DIM), dtype=np.float32)
    to_compute: list[tuple[int, str]] = []
    for i, cycle_id in enumerate(rows["cycle_id"]):
        cache_path = cache_dir / f"{cycle_id}.npy"
        if cache_path.exists():
            embeddings[i] = np.load(cache_path)
        else:
            to_compute.append((i, cycle_id))

    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    batch_indices: list[int] = []
    batch_audio: list[np.ndarray] = []

    def flush() -> None:
        if not batch_audio:
            return
        batch_emb = embed_batch(batch_audio, sample_rate=16000)
        for idx, emb in zip(batch_indices, batch_emb, strict=True):
            embeddings[idx] = emb
            np.save(cache_dir / f"{rows.iloc[idx]['cycle_id']}.npy", emb)
        batch_indices.clear()
        batch_audio.clear()

    rows_by_pos = rows.reset_index(drop=True)
    for i, cycle_id in to_compute:
        row = rows_by_pos.iloc[i]
        record_id = row["record_id"]
        if record_id not in audio_cache:
            audio_cache[record_id] = sf.read(str(audio_dir / f"{record_id}.wav"))
            if len(audio_cache) > 50:  # ponytail: bound memory, drop oldest-ish entry
                audio_cache.pop(next(iter(audio_cache)))
        audio, sr = audio_cache[record_id]
        cycle_audio = extract_cycle(audio, sr, row["start_s"], row["end_s"])
        prepped = prepare_encoder_input(cycle_audio, sr, pad_mode=pad_mode)
        batch_indices.append(i)
        batch_audio.append(prepped)
        if len(batch_audio) >= BATCH_SIZE:
            flush()
    flush()
    return embeddings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D48: E05 frozen-embedding probe under grouped CV.")
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cycles-csv", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--embedding-cache", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--e01-cv-scores", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=CV_SEED)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    cycles_df = pd.read_csv(args.cycles_csv)
    rows = build_trainval_rows(manifest_df, cycles_df)

    print(f"extracting/caching embeddings for {len(rows)} cycles...")
    X = extract_and_cache_embeddings(rows, args.audio_dir, args.embedding_cache)
    y = rows["label4"].to_numpy()
    patient_ids = rows["patient_id"].astype(str).to_numpy()

    folds = make_folds(patient_ids, y, n_splits=args.n_splits, seed=args.seed)
    assert_no_patient_crosses_folds(patient_ids, folds)
    cv_result = run_grouped_cv(X, y, patient_ids, build_pipeline, folds)

    e01_cv = json.loads(args.e01_cv_scores.read_text())
    threshold = e01_cv["macro_f1_mean"] + e01_cv["macro_f1_sd"]
    go_decision = cv_result["macro_f1_mean"] > threshold

    result = {
        "protocol": "E05: frozen AST embeddings -> logistic regression, grouped CV (ADR-0001 D4b)",
        "checkpoint": CHECKPOINT,
        "split_hash": sha256_of_file(args.splits_csv),
        "n_trainval_patients": len(set(patient_ids)),
        "n_trainval_cycles": len(y),
        "seed": args.seed,
        **cv_result,
        "e01_cv_macro_f1_mean": e01_cv["macro_f1_mean"],
        "e01_cv_macro_f1_sd": e01_cv["macro_f1_sd"],
        "e06_go_stop_threshold": threshold,
        "e06_decision": "go" if go_decision else "stop",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "probe_scores.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(
        f"E05 macro_f1={result['macro_f1_mean']:.4f}+/-{result['macro_f1_sd']:.4f} "
        f"vs E01 threshold={threshold:.4f} -> E06 decision: {result['e06_decision'].upper()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

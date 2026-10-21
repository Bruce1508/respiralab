"""D55: zero-shot external validation of frozen E01 and v2 on HF_Lung_V1.

Per ADR-0001 D6 and docs/adr/0002-external-label-mapping.md: no fine-tuning, no
threshold or temperature changes on this data. Reported as its own protocol,
never merged into the ICBHI tables. E01 runs on the full corpus (cheap);
v2 (AST) runs on a stratified random sample (expensive per-item, agreed with
the user given real D48 throughput: ~60 phases/min on this hardware would be
~12h for the full ~47.7k included phases).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor

from respiralab.data.hf_lung_labels import assign_phase_labels, parse_label_file
from respiralab.dsp.features import extract_features
from respiralab.dsp.resample import resample_audio
from respiralab.dsp.segment import extract_cycle
from respiralab.evaluation.bootstrap import patient_bootstrap_ci
from respiralab.evaluation.icbhi_metric import icbhi_score
from respiralab.evaluation.metrics import LABEL_ORDER, score_predictions
from respiralab.inference.predict import TARGET_SAMPLE_RATE, build_frozen_predictor
from respiralab.models.encoder import CHECKPOINT
from respiralab.models.finetune import (
    INDEX_TO_LABEL,
    ASTClassifier,
    RawCycleDataset,
    collate_raw,
    collect_logits,
    select_device,
)

V2_SAMPLE_SIZE = 2000
V2_SAMPLE_SEED = 42
UNFREEZE_LAST_N = 2
BATCH_SIZE = 8
MIN_RESAMPLED_SAMPLES = 64  # below this, extract_features' spectral-contrast bands have no content


def macro_f1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return score_predictions(y_true, y_pred)["macro_f1"]


def _device_prefix(filename: str) -> str:
    if filename.startswith("steth_"):
        return "steth"
    if filename.startswith("trunc_"):
        return "trunc"
    return "other"


def build_external_rows(split_dirs: dict[str, Path]) -> tuple[pd.DataFrame, int]:
    """Parses every *_label.txt under each split dir into one row per included
    (non-excluded) breath phase. Returns (rows_df, n_excluded_phases).
    """
    rows = []
    n_excluded = 0
    for split_name, root in split_dirs.items():
        for label_path in sorted(root.glob("*_label.txt")):
            stem = label_path.name[: -len("_label.txt")]
            wav_path = root / f"{stem}.wav"
            if not wav_path.exists():
                continue
            events = parse_label_file(label_path.read_text())
            for i, phase in enumerate(assign_phase_labels(events)):
                if phase.label4 is None:
                    n_excluded += 1
                    continue
                rows.append(
                    {
                        "record_id": f"{root.name}/{stem}",
                        "phase_id": f"{root.name}/{stem}__{i}",
                        "split": split_name,
                        "device": _device_prefix(stem),
                        "label4": phase.label4,
                        "start_s": phase.start_s,
                        "end_s": phase.end_s,
                    }
                )
    return pd.DataFrame(rows), n_excluded


def stratified_sample(rows: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional-to-natural-frequency sample across label4, no oversampling of
    minority classes (that would misrepresent the true class balance in the report).
    """
    frac = min(1.0, n / len(rows))
    sampled = rows.groupby("label4", group_keys=False)[rows.columns].apply(
        lambda g: g.sample(frac=frac, random_state=seed)
    )
    return sampled.reset_index(drop=True)


def evaluate_e01_external(rows: pd.DataFrame, audio_root: Path, pipeline) -> dict:
    """Full-corpus zero-shot E01 inference: resample -> extract_features -> predict."""
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    features = []
    kept_rows = []
    skipped = 0
    for row in rows.itertuples():
        if row.record_id not in audio_cache:
            if len(audio_cache) > 50:
                audio_cache.pop(next(iter(audio_cache)))
            audio_cache[row.record_id] = sf.read(str(audio_root / f"{row.record_id}.wav"))
        audio, sr = audio_cache[row.record_id]
        try:
            segment = extract_cycle(audio, sr, row.start_s, row.end_s)
            if not np.isfinite(segment).all() or np.allclose(segment, 0.0):
                skipped += 1
                continue
            resampled = resample_audio(segment, sr, TARGET_SAMPLE_RATE)
            if len(resampled) < MIN_RESAMPLED_SAMPLES:
                # ponytail: a handful of HF_Lung annotations have ~1ms phases
                # (annotation rounding, not a real sound event) -- too short for
                # any spectral feature. Skip rather than crash; see
                # reports/external_validation.md's skipped-phase count.
                skipped += 1
                continue
            features.append(extract_features(resampled, TARGET_SAMPLE_RATE))
            kept_rows.append(row)
        except (ValueError, IndexError):
            skipped += 1
            continue

    X = np.stack(features)
    y_true = np.array([r.label4 for r in kept_rows])
    y_pred = pipeline.predict(X)
    record_ids = np.array([r.record_id for r in kept_rows])

    scores = score_predictions(y_true, y_pred, labels=LABEL_ORDER)
    icbhi = icbhi_score(y_true, y_pred, labels=LABEL_ORDER)
    bootstrap = patient_bootstrap_ci(record_ids, y_true, y_pred, macro_f1_metric)
    return {
        "model": "logistic_regression (E01, frozen per reports/model_freeze.md, Gate G4)",
        "n_phases_evaluated": len(y_true),
        "n_phases_skipped_invalid_audio": skipped,
        "macro_f1": scores["macro_f1"],
        "per_class": scores["per_class"],
        "confusion_matrix": scores["confusion_matrix"],
        "icbhi_score": icbhi,
        "bootstrap_macro_f1_by_recording": bootstrap,
    }


def evaluate_v2_external(
    rows_sampled: pd.DataFrame,
    audio_root: Path,
    weights_path: Path,
    device: torch.device,
    feature_extractor: ASTFeatureExtractor,
) -> dict:
    """Zero-shot v2 (E06) inference on a stratified sample -- see module docstring."""
    model = ASTClassifier(unfreeze_last_n=UNFREEZE_LAST_N).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    dataset = RawCycleDataset(rows_sampled, audio_root)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_raw(batch, feature_extractor),
    )
    logits, y_true = collect_logits(model, loader, device)
    y_pred = np.array([INDEX_TO_LABEL[i] for i in logits.argmax(axis=1)])
    record_ids = rows_sampled["record_id"].to_numpy()

    scores = score_predictions(y_true, y_pred, labels=LABEL_ORDER)
    icbhi = icbhi_score(y_true, y_pred, labels=LABEL_ORDER)
    bootstrap = patient_bootstrap_ci(record_ids, y_true, y_pred, macro_f1_metric)
    return {
        "model": "AST bounded fine-tune (E06, frozen per reports/v2_model_freeze.md, Gate G7)",
        "n_phases_evaluated": len(y_true),
        "sample_size_requested": V2_SAMPLE_SIZE,
        "sample_seed": V2_SAMPLE_SEED,
        "macro_f1": scores["macro_f1"],
        "per_class": scores["per_class"],
        "confusion_matrix": scores["confusion_matrix"],
        "icbhi_score": icbhi,
        "bootstrap_macro_f1_by_recording": bootstrap,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D55: zero-shot external validation on HF_Lung_V1.")
    parser.add_argument("--hf-lung-train-dir", required=True, type=Path)
    parser.add_argument("--hf-lung-test-dir", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--e01-cache-dir", required=True, type=Path)
    parser.add_argument("--v2-weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    audio_root = args.hf_lung_train_dir.parent
    rows, n_excluded = build_external_rows(
        {"train": args.hf_lung_train_dir, "test": args.hf_lung_test_dir}
    )
    label_counts = rows["label4"].value_counts().to_dict()

    pipeline = build_frozen_predictor(args.feature_manifest, args.e01_cache_dir)
    e01_result = evaluate_e01_external(rows, audio_root, pipeline)

    rows_sampled = stratified_sample(rows, V2_SAMPLE_SIZE, V2_SAMPLE_SEED)
    device = select_device()
    feature_extractor = ASTFeatureExtractor.from_pretrained(CHECKPOINT)
    v2_result = evaluate_v2_external(rows_sampled, audio_root, args.v2_weights, device, feature_extractor)

    result = {
        "protocol": (
            "zero-shot external validation (ADR-0001 D6, docs/adr/0002-external-label-mapping.md); "
            "no tuning of any kind; separate protocol, never merged with ICBHI tables; "
            "labels are not ground truth (dataset's own Disclaimer)"
        ),
        "n_total_phases": int(len(rows) + n_excluded),
        "n_excluded_rhonchi_stridor_only": n_excluded,
        "n_included_phases": len(rows),
        "label_counts": label_counts,
        "population_and_device_shift": (
            "HF_Lung_V1: adult (TSECC emergency/critical-care + RCW/RCC respiratory-care-ward "
            "residents), devices Littmann 3200 stethoscope ('steth_') and custom HF_Type-1 "
            "multichannel recorder ('trunc_'). ICBHI: different patient population and device mix "
            "entirely (see docs/03_DATA_AND_EVALUATION_CONTRACT.md)."
        ),
        "e01": e01_result,
        "v2": v2_result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    print(f"E01 external macro_f1={e01_result['macro_f1']:.4f} (n={e01_result['n_phases_evaluated']})")
    print(f"v2 external macro_f1={v2_result['macro_f1']:.4f} (n={v2_result['n_phases_evaluated']}, sampled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

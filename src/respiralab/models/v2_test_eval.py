"""D53: ONE-TIME primary-test evaluation of the frozen v2 model (E06) together with
calibrated E01 -- Gate G7's companion exposure, per ADR-0001 D4.

This is the "second exposure of the primary test, post-D29" -- the test was already
opened once at D29 for E01, and its per-class aggregates (via docs/08_FINDINGS or
reports/final_test_results.json) indirectly informed later v2 design choices (D3).
So this cannot be called v2's unbiased first look either; the external dataset (D6)
remains the headline generalization evidence for v2.

No subsequent tuning may follow this report. No number here may ever be cited to
justify a v2 design change.
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
from respiralab.evaluation.bootstrap import patient_bootstrap_ci
from respiralab.evaluation.calibration import expected_calibration_error, softmax
from respiralab.evaluation.icbhi_metric import icbhi_score
from respiralab.evaluation.metrics import LABEL_ORDER, score_predictions
from respiralab.models.baseline import load_split_features, load_split_patient_ids
from respiralab.models.encoder import CHECKPOINT
from respiralab.models.finetune import (
    INDEX_TO_LABEL,
    ASTClassifier,
    RawCycleDataset,
    collate_raw,
    collect_logits,
    git_commit,
    select_device,
)
from respiralab.models.traditional import build_pipeline

UNFREEZE_LAST_N = 2  # must match D52's frozen architecture
BATCH_SIZE = 8


def macro_f1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return score_predictions(y_true, y_pred)["macro_f1"]


def build_test_rows(manifest_df: pd.DataFrame, cycles_df: pd.DataFrame) -> pd.DataFrame:
    """Same shape as finetune.build_trainval_rows, but for the primary test split."""
    rows = manifest_df[manifest_df["split"] == "test"].copy()
    if rows.empty:
        raise ValueError("no feature_manifest rows for split='test'")
    bounds = cycles_df.set_index("cycle_id")[["record_id", "start_s", "end_s"]]
    rows = rows.join(bounds, on="cycle_id", rsuffix="_bounds")
    if rows["start_s"].isna().any():
        missing = rows.loc[rows["start_s"].isna(), "cycle_id"].tolist()
        raise ValueError(f"cycle_id(s) missing from cycles.csv: {missing[:5]}")
    return rows


def evaluate_e01_test(manifest_df: pd.DataFrame, cache_dir: Path, calibration_temperature: float) -> dict:
    """Refit the frozen E01 pipeline on the original train split (84 patients, Gate G4
    spec) and evaluate exactly once on test -- reproduces D29's macro_f1_test bit-for-bit
    (E01 has zero training stochasticity, D20), then applies D45's temperature (fit on
    OOF CV predictions, never on test) for a calibrated ECE-on-test reading only.
    """
    X_train, y_train = load_split_features(manifest_df, "train", cache_dir)
    X_test, y_test = load_split_features(manifest_df, "test", cache_dir)
    patient_ids_test = load_split_patient_ids(manifest_df, "test")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    logits = pipeline.decision_function(X_test)
    logits = logits[:, [list(pipeline.classes_).index(c) for c in LABEL_ORDER]]

    scores = score_predictions(y_test, y_pred, labels=LABEL_ORDER)
    icbhi = icbhi_score(y_test, y_pred, labels=LABEL_ORDER)
    bootstrap = patient_bootstrap_ci(patient_ids_test, y_test, y_pred, macro_f1_metric)

    ece_before = expected_calibration_error(softmax(logits), y_test)
    ece_after = expected_calibration_error(softmax(logits / calibration_temperature), y_test)

    return {
        "model": "logistic_regression (E01, frozen per reports/model_freeze.md, Gate G4)",
        "n_test_cycles": len(y_test),
        "macro_f1_test": scores["macro_f1"],
        "per_class_test": scores["per_class"],
        "confusion_matrix_test": scores["confusion_matrix"],
        "icbhi_score_test": icbhi,
        "bootstrap_macro_f1": bootstrap,
        "calibration_temperature": calibration_temperature,
        "ece_before": ece_before["ece"],
        "ece_after": ece_after["ece"],
    }


def evaluate_v2_test(
    rows_test: pd.DataFrame,
    audio_dir: Path,
    weights_path: Path,
    calibration_temperature: float,
    device: torch.device,
    feature_extractor: ASTFeatureExtractor,
) -> dict:
    """Load the frozen v2 (E06) weights and evaluate exactly once on test."""
    model = ASTClassifier(unfreeze_last_n=UNFREEZE_LAST_N).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    dataset = RawCycleDataset(rows_test, audio_dir)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_raw(batch, feature_extractor),
    )
    logits, y_true = collect_logits(model, loader, device)
    y_pred = np.array([INDEX_TO_LABEL[i] for i in logits.argmax(axis=1)])
    patient_ids_test = rows_test["patient_id"].astype(str).to_numpy()

    scores = score_predictions(y_true, y_pred, labels=LABEL_ORDER)
    icbhi = icbhi_score(y_true, y_pred, labels=LABEL_ORDER)
    bootstrap = patient_bootstrap_ci(patient_ids_test, y_true, y_pred, macro_f1_metric)

    ece_before = expected_calibration_error(softmax(logits), y_true)
    ece_after = expected_calibration_error(softmax(logits / calibration_temperature), y_true)

    return {
        "model": "AST bounded fine-tune (E06, frozen per reports/v2_model_freeze.md, Gate G7)",
        "n_test_cycles": len(y_true),
        "macro_f1_test": scores["macro_f1"],
        "per_class_test": scores["per_class"],
        "confusion_matrix_test": scores["confusion_matrix"],
        "icbhi_score_test": icbhi,
        "bootstrap_macro_f1": bootstrap,
        "calibration_temperature": calibration_temperature,
        "ece_before": ece_before["ece"],
        "ece_after": ece_after["ece"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D53: ONE-TIME primary-test evaluation (frozen v2 + calibrated E01)."
    )
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--cycles-csv", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path, help="E01 feature cache (D13)")
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--splits-csv", required=True, type=Path)
    parser.add_argument("--e01-calibration-json", required=True, type=Path)
    parser.add_argument("--v2-freeze-json", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    manifest_df = pd.read_csv(args.feature_manifest)
    cycles_df = pd.read_csv(args.cycles_csv)
    rows_test = build_test_rows(manifest_df, cycles_df)

    e01_calibration = json.loads(args.e01_calibration_json.read_text())
    v2_freeze = json.loads(args.v2_freeze_json.read_text())

    e01_result = evaluate_e01_test(manifest_df, args.cache_dir, e01_calibration["temperature"])

    device = select_device()
    feature_extractor = ASTFeatureExtractor.from_pretrained(CHECKPOINT)
    weights_path = Path(v2_freeze["final_model"]["weights_path"])
    v2_result = evaluate_v2_test(
        rows_test, args.audio_dir, weights_path, v2_freeze["calibration"]["temperature"], device, feature_extractor
    )

    result = {
        "protocol": (
            "second exposure of the primary test, post-D29; design informed by D29 aggregates "
            "(ADR-0001 D4). Run exactly once. No subsequent tuning. External dataset (D6) remains "
            "v2's headline generalization evidence."
        ),
        "git_commit": git_commit(),
        "split_hash": sha256_of_file(args.splits_csv),
        "n_test_patients": int(pd.Series(rows_test["patient_id"]).nunique()),
        "e01": e01_result,
        "v2": v2_result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    print(f"E01 macro_f1_test={e01_result['macro_f1_test']:.4f} (reproduces D29)")
    print(f"v2 (E06) macro_f1_test={v2_result['macro_f1_test']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

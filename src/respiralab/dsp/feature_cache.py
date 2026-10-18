from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from respiralab.dsp.features import extract_features

PREPROCESS_CONFIG = {
    "target_sample_rate": 4000,
    "n_mfcc": 13,
    "n_contrast_bands": 3,
    "feature_schema_version": 1,
}


def preprocess_config_hash() -> str:
    payload = json.dumps(PREPROCESS_CONFIG, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def feature_vector_hash(vec: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(vec, dtype=np.float64).tobytes()).hexdigest()


@dataclass
class CacheRow:
    cycle_id: str
    record_id: str
    patient_id: str
    split: str
    label4: str
    input_sha256: str
    preprocess_config_hash: str
    feature_hash: str
    cache_path: str


def cache_one_cycle(
    audio: np.ndarray,
    sample_rate: int,
    cycle_id: str,
    record_id: str,
    patient_id: str,
    split: str,
    label4: str,
    input_sha256: str,
    cache_dir: Path,
) -> CacheRow:
    vec = extract_features(audio, sample_rate)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"{cycle_id}.npy"
    np.save(cache_dir / rel_path, vec)
    return CacheRow(
        cycle_id=cycle_id,
        record_id=record_id,
        patient_id=patient_id,
        split=split,
        label4=label4,
        input_sha256=input_sha256,
        preprocess_config_hash=preprocess_config_hash(),
        feature_hash=feature_vector_hash(vec),
        cache_path=rel_path,
    )


def write_manifest(rows: list[CacheRow], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(CacheRow)]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(vars(row))

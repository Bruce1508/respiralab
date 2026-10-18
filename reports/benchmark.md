# Feature extraction benchmark (D13)

## Machine
- Python 3.11.15, macOS-26.6.2-arm64-arm-64bit, arm64 (CPU-only, no GPU used — DSP feature extraction is CPU-bound)

## Workload
Full D08→D09→D13 pipeline over the entire real dataset: extract_cycle (crop) →
resample_audio (native rate → 4000Hz) → cache_one_cycle (extract_features +
hash + save .npy), one call per cycle.

## Results
| metric | value |
|---|---|
| cycles processed | 6898 (100% of dataset) |
| total elapsed | 27.7 s |
| mean per cycle | 4.01 ms |
| cache size on disk | 27 MB (`artifacts/features_cache/`, gitignored — regeneratable) |
| manifest size | 1.6 MB (`manifests/feature_manifest.csv`, committed) |

## Determinism check
Re-ran feature extraction (full pipeline, independent cache directory) on a
random 20-cycle sample (`random_state=1`) and compared each `feature_hash`
against the original manifest: **0 mismatches**. Confirms identical inputs +
config always produce identical feature hashes (reproducibility contract),
not merely asserted by the unit tests on synthetic data.

## Interpretation
~4ms/cycle on CPU means the full dataset's feature extraction fits
comfortably within a single interactive session; no need for GPU
acceleration, batching infrastructure, or an MLflow-style async pipeline at
this scale (docs/02_ARCHITECTURE.md: "add MLflow only if logging becomes
painful" — it hasn't).

# D55 — Zero-shot external validation on HF_Lung_V1 (Gate G7 companion, ADR-0001 D6)

**This is a separate protocol, not comparable to any ICBHI table in this repository.** HF_Lung_V1
labels breath phases, not whole respiratory cycles; its population and recording devices are
different from ICBHI's; and the dataset's own `Disclaimer` states its labels are "not ground
truth." No fine-tuning, threshold, or temperature change was made on this data — both models are
exactly as frozen at Gate G4 (E01) and Gate G7 (v2/E06).

## Corpus (per docs/adr/0002-external-label-mapping.md)

| | Count |
|---|---:|
| Total breath phases (I/E) | 52,444 |
| Excluded (Rhonchi/Stridor-only, no D/Wheeze) | 4,691 (8.9%) |
| **Included in primary mapping** | **47,753** |
| &nbsp;&nbsp;normal | 25,382 (53.2%) |
| &nbsp;&nbsp;crackle | 15,508 (32.5%) |
| &nbsp;&nbsp;wheeze | 6,496 (13.6%) |
| &nbsp;&nbsp;both | 367 (0.77%) |

**Population/device shift:** HF_Lung_V1 is adult-sourced (TSECC emergency/critical-care patients +
RCW/RCC respiratory-care-ward residents) recorded with a Littmann 3200 digital stethoscope
(`steth_`) and a custom multichannel acoustic device (`HF_Type-1`, `trunc_`) — both different
from ICBHI's patient population and device mix.

## Results

| Model | Scope | n evaluated | macro-F1 | 95% CI (by recording) | ICBHI Score |
|---|---|---:|---:|---|---:|
| E01 (frozen, Gate G4) | full corpus | 47,749 (4 skipped — degenerate ~1ms phases) | **0.2165** | [0.211, 0.222] | 0.2534 |
| v2 / E06 (frozen, Gate G7) | stratified sample, seed 42 | 2,000 | **0.3025** | [0.283, 0.320] | 0.4913 |

v2 still outperforms E01 zero-shot (0.3025 vs 0.2165), consistent with the ICBHI findings (D53:
0.5208 vs 0.2861) — the improvement direction holds under real distribution shift, though both
models drop sharply from their ICBHI numbers, as expected for a true external-generalization test.

The confidence intervals above are **grouped by recording (`record_id`), not by patient** —
HF_Lung_V1's filenames give no verified per-patient ID (see `docs/adr/0002-external-label-mapping.md`'s
same-date-token caveat), so this is the finest honest grouping available, not a patient-level
bootstrap like D53's ICBHI numbers.

### Per-class detail

| Class | E01 recall (full corpus) | v2 recall (sample) |
|---|---:|---:|
| normal | 0.321 | 0.796 |
| crackle | 0.139 | 0.055 |
| wheeze | 0.283 | 0.507 |
| both | 0.439 (n=367) | 0.067 (n=15) |

**`both`'s numbers are not reliable evidence either direction** — E01's "high" both-recall
(0.439) comes with precision of 0.008 (it's essentially guessing `both` very often and getting
lucky on the rare true positives), and v2's sample contains only 15 true `both` phases out of
2,000 — far too few for a meaningful recall estimate. Neither number should be read as "E01 is
better at `both` than v2 here."

**v2's crackle recall (0.055) is the standout honest weak point**, much lower than its ICBHI
recall and lower than E01's own crackle recall on this same external data. This is reported
plainly, not smoothed over, and is flagged as a concrete v2.1 investigation candidate — possible
causes include the breath-phase (short, often <1s) vs. ICBHI's whole-cycle unit mismatch, or
device-specific crackle acoustics the fine-tune never saw.

## Known limits of this protocol (do not treat as ICBHI-equivalent benchmark numbers)

- **Unit-of-analysis mismatch:** a "phase" here is one inhale or one exhale, not a full
  respiratory cycle like ICBHI's. Models were trained/frozen on whole-cycle inputs; phases are
  typically shorter.
- **v2's sample is 2,000 of 47,753 included phases (4.2%)**, stratified proportionally by class —
  not oversampled for `both`, so its already-tiny natural frequency (0.77%) carries through as an
  even tinier absolute count (15) in the sample. E01 ran on the full corpus.
- **Labels are not ground truth** (dataset's own `Disclaimer`): a single annotator per file, with
  only "good agreement," not inter-rater-verified ground truth.
- **Four E01 phases were skipped** for being under ~1ms (certain annotation-rounding artifacts,
  not real sound events) — see `src/respiralab/models/external_eval.py`'s `MIN_RESAMPLED_SAMPLES`
  guard.

## Reproduction

```bash
uv run python -m respiralab.models.external_eval \
  --hf-lung-train-dir data/external/HF_Lung_V1_train \
  --hf-lung-test-dir data/external/HF_Lung_V1_test \
  --feature-manifest manifests/feature_manifest.csv \
  --e01-cache-dir artifacts/features_cache \
  --v2-weights results/v2_frozen/v2_model_state_dict.pt \
  --out reports/external_validation.json
```

Full numeric output: `reports/external_validation.json`.

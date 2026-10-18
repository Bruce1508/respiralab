# RespiraLab model card (D34, Gate G5)

## What this is — and is not

This model estimates **sound-event labels** (Normal / Crackle / Wheeze /
Both) for expert-annotated respiratory cycles from the ICBHI 2017 dataset.
**It is a research/educational demonstration, not a diagnostic device.**
It does not detect asthma, COPD, pneumonia, or any disease. A sound label
is not a clinical finding. No output from this model should inform a
medical decision.

## Model

- **Frozen model:** E01 — `StandardScaler` (train-only fit) →
  `LogisticRegression(class_weight="balanced")`. Frozen at commit `cdc0b4f`
  (`reports/model_freeze.md`), Gate G4 PASS.
- **Alternatives considered and rejected:** RBF SVM (E02) — statistically
  indistinguishable from E01 with val-set selection bias (D21); compact CNN
  (E03) — underperforms on macro-F1 and fails completely on `both`
  (recall=0.0, D27).

## Performance (primary, patient-disjoint protocol)

| split | macro_f1 | icbhi_score | both_recall | wheeze_recall |
|---|---:|---:|---:|---:|
| val (dev) | 0.3782 | 0.4656 | 0.0441 | 0.5714 |
| **test (final, one-time, D29)** | **0.2861** | **0.3762** | **0.3288** | **0.1038** |

Test macro-F1 is honestly lower than val — a real generalization gap, not
adjusted. Per-class behavior shifts between val and test for the two
smallest-support classes (`wheeze`: 106 test cycles, `both`: 73 test
cycles); plausibly split-composition noise given the small support, not a
claimed systematic effect.

Separately, the **official ICBHI split** (not patient-disjoint — patients
156/218 appear in both its train and test) gives macro_f1=0.3366 (D30).
**This number must never be combined with the primary-split result above.**

## Known limitations (all independently verified, not assumed)

1. **`both`-class recall is weak across every model tried** — 0.044 (E01),
   0.147 (E02), 0.0 (CNN) on val; the project's clearest, most consistent
   failure mode (D21, D27, D33).
2. **19.2% of ICBHI recordings have measurable clipping** (worst case
   48.2% of samples), identified during DSP review (D14) — not corrected,
   carried forward as a known data-quality caveat.
3. **Noise sensitivity:** macro-F1 degrades at every tested SNR vs clean
   (worst: -0.0934 at 0dB, D31). Supported-use claim is therefore limited
   to recordings comparable to ICBHI's own acquisition conditions, not
   arbitrary noisy consumer-microphone audio.
4. **ICBHI Score has a Normal-specificity blind spot** (D21, D27, D31): a
   model (or noise condition) that biases toward predicting Normal scores
   *higher* on ICBHI Score while performing *worse* on macro-F1 and
   per-class recall. Always read ICBHI Score alongside macro-F1, never
   alone.
5. **Class-weighting matters for the minority class, not for the headline
   metric** (D33): removing `class_weight="balanced"` barely moves macro-F1
   (within noise) but completely zeroes `both`-class recall.
6. **Error analysis (D32):** of 144 misclassified wheeze/both test cycles
   reviewed, errors cluster into two non-random types — missed-abnormal-
   to-Normal, and confusion between adjacent abnormal classes — not random
   noise. Device/site error counts are raw, not rate-normalized; no device
   effect is claimed.
7. **Dataset scope:** 126 patients, 920 recordings, 4 heterogeneous
   acquisition devices and sample rates (4k/10k/44.1kHz) — results do not
   generalize to other microphones, populations, or recording protocols
   without re-validation.

## Intended use

Research and educational demonstration of a reproducible, leakage-checked
sound-event classification pipeline. Suitable for: ML engineering review,
methodology inspection, reproducibility verification. **Not suitable for:**
any clinical, diagnostic, or patient-facing decision; unattended long-
recording analysis; any claim of disease detection.

## Gate G5 go/no-go checklist (`planning/QUALITY_GATES.md`)

1. Numbers from actual execution artifacts? Yes — every number above cites
   a specific prior commit/report (D14, D21, D27, D29-D33).
2. Trained for sound labels, not medical disease? Yes, explicitly stated
   above and throughout.
3. Any leakage between sets? No — re-affirms D28's frozen checklist; D29's
   test evaluation was one-time and is now closed.
4. Reproducible without notebook state? Yes — no notebooks in this repo.
5. Defensible limitations to an ML engineer? Yes — 7 limitations listed
   above, none hidden.

**Gate G5 decision: PASS.** No diagnosis claims are made; limitations are
explicit. Product phase (D35+) may proceed.

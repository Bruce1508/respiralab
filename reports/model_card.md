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

---

# v2 model card (D56/D57, Gate G8)

## What this is — and is not

Same scope as v1 above: **sound-event labels** (Normal/Crackle/Wheeze/Both)
for expert-annotated respiratory cycles. **Still not a diagnostic device.**
v2 changes the model, not the claim.

## Model

- **Frozen model:** E06 — pretrained AST encoder (`MIT/ast-finetuned-audioset-10-10-0.4593`,
  AudioSet-pretrained), last 2 transformer blocks + layernorm + a new linear
  head bounded-fine-tuned (`unfreeze_last_n_layers=2`, `lr=1e-5`,
  `final_epochs=2`). Frozen at commit `79ab2d8` (`reports/v2_model_freeze.md`,
  weights sha256 `e3ddc62d...`), Gate G7 PASS.
- **Alternatives considered and rejected:** frozen-embedding probe (E05,
  macro_f1=0.4448±0.0226) — E06 clears E05's own mean+1sd bar
  (0.4791±0.0401 > 0.4674); repeat-pad (E08, macro_f1=0.4243±0.0298) loses
  to zero-pad (D51); balanced_sampling (E07, both_recall=0.3683±0.1067 vs
  reference 0.2634±0.1516) clearly wins on the minority class but was **not**
  fused into v2 — single-factor discipline per ADR-0001, flagged as a v2.1
  candidate rather than bolted on without its own isolated validation.

## Performance (primary, patient-disjoint protocol — kept separate from v1's table above)

| protocol | model | macro_f1 | 95% CI | source |
|---|---|---:|---:|---|
| grouped CV (5-fold, 3 seeds) | E06 | 0.4791 | ±0.0401 | `results/E06/grid_scores.json` |
| grouped CV (5-fold, 3 seeds) | E01 (recomputed reference) | 0.3654 | ±0.0330 | `results/E01_cv/cv_scores.json` |
| **primary test (one-time, second exposure, D53)** | **E06 (v2)** | **0.5208** | **[0.416, 0.558]** | `reports/v2_test_results.json` |
| primary test (one-time, second exposure, D53) | E01 (refit reference) | 0.2861 | [0.233, 0.316] | `reports/v2_test_results.json` |
| external zero-shot (HF_Lung_V1, no tuning, D55) | v2 (2,000-phase stratified sample) | 0.3025 | — | `reports/external_validation.json` |
| external zero-shot (HF_Lung_V1, no tuning, D55) | E01 (full corpus) | 0.2165 | — | `reports/external_validation.json` |

v2 beats E01 on every protocol above — CV, primary test, and external —
and the primary test score (0.5208) sits consistent with (slightly above)
v2's own CV estimate, not an isolated fluke. The external numbers are a
**separate protocol table**, never merged with the ICBHI rows above
(different unit of analysis: breath phase, not whole cycle — ADR-0002).

## Known limitations (all independently verified, not assumed)

1. **Calibration regresses on test, opposite of v1's direction** — v2's
   OOF-fit temperature (1.3288) *increases* test ECE (0.0464→0.0949),
   while E01's (fit the same way) *decreases* it (0.1273→0.0495), D53.
   Argmax predictions and macro-F1 are unaffected by temperature; only
   confidence calibration is implicated. Flagged as a v2.1 investigation,
   not explained away.
2. **External crackle recall is weak** — 0.055 on HF_Lung_V1 (D55), lower
   than E01's 0.139 on the same data, despite v2 winning on overall
   macro-F1. A real, disclosed weak point, not hidden behind the headline
   number.
3. **External validation used a stratified sample for v2, not the full
   corpus** (2,000 of ~45k phases, seed 42, D55) — real throughput
   benchmarking showed the full run would take ~12 hours; E01 (cheap) still
   ran on the full corpus. The external macro-F1 therefore carries slightly
   more sampling variance for v2 than for E01.
4. **Weights are not committed to git** — 328.86MB exceeds GitHub's 100MB
   limit (D55). Served from Hugging Face Hub (`bruce1508/respiralab-v2`),
   downloaded and sha256-verified against `results/v2_frozen/v2_freeze.json`
   before every load; a hash mismatch is a hard refusal (D56).
5. **OOD gate is a Mahalanobis heuristic on v2's own embeddings, fit on
   5,757 train+val cycles only** (threshold=42.136 at p99, D56) — it rejects
   inputs far from ICBHI's own recording conditions, but passing the gate is
   not itself evidence of correctness; it only screens out clearly
   off-distribution input (e.g. non-respiratory audio).
6. **All of v1's limitations still apply where unchanged** — dataset scope
   (126 patients, 4 devices), ICBHI Score's Normal-specificity blind spot;
   noise robustness and error-analysis studies (D31/D32) were run against
   E01 and have not been repeated against v2.

## Intended use

Same as v1: research/educational demonstration. **Not suitable for:** any
clinical, diagnostic, or patient-facing decision; any claim of disease
detection. v2 additionally requires network access (to Hugging Face Hub)
at first load, unlike v1's git-committed, fully offline frozen pipeline.

## Gate G8 go/no-go checklist (`planning/QUALITY_GATES.md`)

1. One-time second-exposure test report? Yes — `reports/v2_test_results.json`
   (D53), run exactly once, no subsequent tuning.
2. External zero-shot report (or documented blocker)? Yes —
   `reports/external_validation.json`/`.md` (D55), population/device shift
   stated explicitly, no tuning on external data.
3. Hash-verified serving? Yes — `src/respiralab/inference/v2_predict.py`
   (D56), verified end-to-end against the real HF Hub repo.
4. Updated model card? Yes — this section.
5. Fresh clone? Yes — `uv sync && uv run pytest -q` passes 247/247 on a
   clone with no ICBHI data or HF credentials present (the 11 v2-serving
   tests are fully synthetic/mocked; only the two manual real-data/real-HF
   verifications in D56 and D57 required actual data and network access).

**Gate G8 decision: PASS.** v1 and v2 results are reported in separate
tables throughout this card and the final report; no diagnosis claim is
made for either model.

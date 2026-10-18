# ADR-0001 — v2: pretrained encoder at 16 kHz, grouped-CV selection, calibration, external validation

- **Status:** Accepted (planning). No v2 code exists yet; tasks D44–D57 implement it.
- **Date:** 2026-10-18
- **Supersedes:** nothing. v1.0 (E01, D01–D42, Gates G1–G6) stays the published, frozen baseline.
- **Required by:** `docs/03_DATA_AND_EVALUATION_CONTRACT.md` header and `AGENTS.md` — "any change in split, label definition, cutoff/sample rate or metric needs an explicit ADR and invalidates prior comparisons if incompatible." v2 changes the sample rate and the model-selection protocol.

## Context (numbers cite committed artifacts)

- Frozen v1 model: E01 logistic regression on 46 handcrafted features, 4 kHz, bandpass 100–1800 Hz (`reports/model_freeze.md`, preprocess hash `2702dab77672cc55`, split hash `38e11a7f…68da`).
- v1 primary test, opened once at D29 (`reports/final_test_results.json`): macro-F1 **0.2861**; per-class recall normal 0.437, crackle 0.407, wheeze 0.104, both 0.329.
- v1 validation (`results/E01/dev_scores.json`): macro-F1 0.3782; recall normal 0.358, crackle 0.698, wheeze 0.571, both 0.044.
- The val→test swings (wheeze 0.571→0.104, both 0.044→0.329) on 21-patient holdouts are consistent with high variance from small patient counts. A single 21-patient val split is too noisy to select a v2 model on.
- CNN E03 (4 kHz log-Mel, zero-padded to 6.0 s) lost to E01 on val (0.3157 vs 0.3782) with both-recall 0.0 (`reports/model_comparison.md`).
- Native ICBHI sample rates are 4 000 / 10 000 / 44 100 Hz (D09). 4 kHz sources carry no content above 2 kHz.

## Decisions

### D1. v2 is a new experiment family (E05+), not a revision of v1
v1 code paths, configs, hashes and reports stay untouched. v2 numbers are **not** placed in the same table as v1 single-split val numbers. E01 is re-run under the v2 protocol (D44) to give a like-for-like reference.

### D2. Encoder input: 16 kHz, common band ≤ 1800 Hz
- Resample to 16 kHz only because the AudioSet-pretrained candidates (AST, BEATs) take 16 kHz input. Upsampling a 4 kHz source adds no information. **D47 must confirm the chosen checkpoint's input rate. If it differs, amend this ADR before extracting embeddings.**
- All v2 inputs are low-passed to the v1 upper edge (1800 Hz, below the 2 kHz Nyquist of the 4 kHz sources) before encoding. Otherwise, energy above 2 kHz exists only for 10k/44.1k devices and becomes a device shortcut (device/site confounds: `docs/07_SOURCE_VERIFICATION.md` item 4).
- Rejected: full-band 16 kHz for high-rate sources. It might help crackles, but it confounds device with label. It is deferred, and if it is ever tried it must be a single ablation with a per-device breakdown.

### D3. Model selection: 5-fold patient-grouped, label-stratified CV over train ∪ val
- Pool the 105 train+val patients of the frozen primary split and fold them by patient. The 21 test patients are excluded from every fold.
- All v2 model, encoder, ablation and temperature choices are made on CV mean ± sd across folds. Patient-group bootstrap is used for intervals, as in the contract.
- Fold assignment is deterministic (seed 42) and its hash is recorded next to the split hash.

### D4. Primary-test exposure policy
- The primary test was opened at D29. Its per-class results are known, and they informed D3 above (the protocol, not a model). So a v2 primary-test number cannot be called a fully unbiased first look.
- v2 is evaluated on the primary test **once**, after the G7 freeze (D53), together with calibrated E01. The report labels it *"second exposure of the primary test, post-D29; design informed by D29 aggregates"*.
- No v2 decision may cite primary-test numbers. After D53 the test is closed for good.
- The **external dataset (D6) is the headline generalization evidence for v2**, because it is the only holdout nobody has looked at.

### D4b. Pre-registered experiments (one factor at a time, all on D3 folds)
| ID | What | Go/stop rule |
|---|---|---|
| E05 | Frozen encoder embeddings → logistic regression (same head family as E01) | Fine-tune (E06) only if E05 CV mean macro-F1 > E01 CV mean + 1 fold-sd. Otherwise skip E06 and report a negative result. |
| E06 | Bounded fine-tune of the same encoder, 3 seeds [13, 21, 42] | Same bounded-budget rule as `04_EXPERIMENTS.md` |
| E07a/b/c | Class-balanced sampling / focal loss / train-only mixup, each vs a class-weighted reference | Pick by CV macro-F1, with both-recall reported. Mixup: both parents must be train-fold patients (contract leakage assert). |
| E08 | Repeat-pad vs zero-pad (v1 used zero-pad, `models/dataset.py::pad_or_crop`) | Single factor on the best E05–E07 config |

### D5. Calibration
Fit temperature scaling on held-out fold predictions only, never on test. Report ECE (10 equal-width bins on max-probability) and a reliability diagram, before and after scaling, for E01 and the frozen v2 model.

### D6. External validation (zero-shot)
- Candidates: SPRSound (paediatric) or HF_Lung. **None has been downloaded.** The user acquires the data and records its license or terms first (D54).
- The external label sets do not match Normal/Crackle/Wheeze/Both one-to-one. The mapping, any exclusions and the cycle or segment unit are fixed in **ADR-0002 before any prediction is run**.
- No fine-tuning, threshold or temperature changes on external data. It is reported as its own protocol, never merged into the ICBHI tables, with population and device shift stated.

### D7. Serving
- A fine-tuned encoder cannot be refit at Streamlit startup the way E01 is. Weights become a versioned artifact (HF Hub) loaded by sha256, and the app refuses to load on a hash mismatch. The weight license must permit redistributing a derivative.
- Add an input-quality/OOD gate fit on train data only. It warns or rejects, and does not silently predict.
- Rejected: a separate FastAPI service. `02_ARCHITECTURE.md` says to avoid it until needed, and Streamlit can call `respiralab.inference` in-process.

### D8. Compute
CPU / Apple-silicon (MPS) local only, with a bounded budget. If E06 does not fit the budget, stop at E05 and document it (`DECISION_RULES.md`: "Compute budget exhausted").

## Consequences

- New gates: **G7** (v2 freeze, D52) and **G8** (v2 release, D57) in `planning/QUALITY_GATES.md`.
- v1 must keep reproducing bit-for-bit: preprocess hash `2702dab77672cc55` and `results/E01/dev_scores.json`. This is checked in D46.
- No literature number is cited until verified. The handoff's "≈60% ICBHI Score for pretrained encoders" is unverified. D47 either sources it in `07_SOURCE_VERIFICATION.md` or drops it.
- Still not a diagnostic device. v2 changes nothing about the sound-event-only claim.

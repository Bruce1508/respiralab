# D28 — Model freeze and pre-test protocol declaration (Gate G4)

**Frozen at commit:** `cdc0b4f` on `main`, https://github.com/Bruce1508/respiralab
**Frozen at date:** 2026-10-18
**Primary split hash (frozen, unchanged since D07):** `38e11a7fa958b408af835e4dd89ba31364482b7734042606f127ef92900068da`
**Feature preprocess hash (frozen, D13):** `2702dab77672cc55`

## Declared frozen model

**E01 — logistic regression** (`src/respiralab/models/traditional.py`, commit `07fff5a`):
- Pipeline: `StandardScaler()` (fit train-only, verified by `test_scaler_fits_only_on_train_not_dev`) → `LogisticRegression(class_weight="balanced")`, scikit-learn defaults otherwise (solver=`lbfgs`).
- Training seed: 42 (recorded for reproducibility; D20 confirmed `lbfgs` does not actually consume `random_state`, so this run has zero training stochasticity — reproducibility is proven by exact reruns, not by seed control).
- Decision rule: hard 4-class argmax (`predict()`). No probability threshold is tuned or frozen — this project's primary task is 4-class, not a binary Normal/Abnormal threshold decision.
- "Frozen" here means: this exact code + config + split, refit deterministically, reproduces the committed `results/E01/dev_scores.json` bit-for-bit (already verified in D16 by diffing two independent reruns). No separate pickled model artifact is stored, since there is no training stochasticity to protect against for this pipeline.

**Why E01, not E02 (SVM) or E03 (CNN)** — unchanged from D21/D27:
- D21 (Gate G3): E02's 0.009 macro-F1 edge over E01 is inside both models' patient-bootstrap CI and carries val-set selection bias from its 9-config search that E01 doesn't have.
- D27: CNN (best of 3 seeds) underperforms both classical models on macro-F1 (0.3157 vs 0.3782) and fails completely on the `both` class (recall=0.0). Retained as a documented comparison, not a freeze candidate.

## Frozen evaluation protocol for D29

- Metric: macro-F1 (primary), ICBHI Score + per-class precision/recall/F1 + confusion matrix (secondary), exactly as implemented in `src/respiralab/evaluation/metrics.py` and `icbhi_metric.py` — unchanged since D18/D19.
- Test set: the primary `test` split from `manifests/splits_primary.csv` (21 patients, 134 recordings, 1141 cycles per D07), `test_locked=true` until this freeze. D29 may now open it, exactly once, with no subsequent tuning.
- No further changes to model config, feature pipeline, split, or metric definitions are permitted after this point without an explicit ADR (per `AGENTS.md`'s critical scientific constraints) — doing so would invalidate the D29 result.

## Gate G4 go/no-go checklist (`planning/QUALITY_GATES.md`)

1. **Are all numbers derived from actual execution artifacts, not stated ambitions?** Yes — every number above cites a specific committed file/commit (D16/D17/D20/D21/D23-D27), nothing restated from memory.
2. **Is the underlying model trained for sound labels, not medical disease?** Yes — 4-class sound-event labels (normal/crackle/wheeze/both) only; no diagnosis claim anywhere in this pipeline.
3. **Are any subjects, feature fit operations, or augmentations leaking between sets?** No — scaler is train-only (D16, directly tested), SVM search is val-only selection (D17), CNN best-checkpoint selection is val-only (D24, directly tested), augmentation is hard-guarded to train patients only (D25, raises if violated).
4. **Can a second developer reproduce the evidence without copying notebook state?** Yes — every result comes from a committed CLI command + config + fixed seed; no notebooks exist in this repo.
5. **Would showing this to an ML engineer produce defensible answers about limitations?** Yes — `both`-class near-zero recall across every model (worst: CNN at 0.0), 19.2% recording clipping (D14), and CNN's misleading-if-read-alone ICBHI Score (D27) are all explicitly documented, not hidden.

**Gate G4 decision: PASS.** Model selection (E01 logistic regression) is frozen. D29 (one-time final test evaluation) may proceed.

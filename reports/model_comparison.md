# D27 — CNN vs classical ML, identical validation protocol

All three candidates evaluated on the same `val` split (21 patients, 1108 cycles,
`split_hash=38e11a7f...`). No test-set data used. CNN uses its best seed (13) from
D26's 3-seed sweep; E01/E02 are D16/D17's committed results.

## Comparison table

| experiment_id | commit | seed | model | macro_f1_val | icbhi_score_val | both_recall_val | notes |
|---|---|---|---|---:|---:|---:|---|
| E01 | 07fff5a | 42 | logistic_regression | 0.3782 | 0.4656 | 0.0441 | train-only scaler, class_weight=balanced |
| E02 | dd6427a | 42 | svm_rbf (C=1.0, gamma=0.01) | 0.3872 | 0.4763 | 0.1471 | val-selected over 9 configs, class_weight=balanced |
| E03 | 5aaeddd | 13 (best of 3) | compact_cnn | 0.3157 | 0.5083 | 0.0 | 6 epochs, no class weighting |

macro_f1_test_final / icbhi_score_test_final: not applicable — test stays locked
until Gate G4 (D28) per `planning/QUALITY_GATES.md`.

## Honest read

**CNN underperforms both classical baselines on macro-F1** (0.3157 vs
0.3782/0.3872) and **completely fails on the `both` minority class**
(recall=0.0, support=68) — worse than both E01 (0.044) and E02 (0.147). The
most likely cause, not yet tested: unlike E01/E02, the CNN was trained with
plain `CrossEntropyLoss`, no `class_weight="balanced"` equivalent. This is a
documented gap, not something D27 fixes.

**CNN's ICBHI Score is the highest of the three (0.5083) despite being the
worst model by macro-F1 and per-class recall** — this is the same blind spot
D21 already flagged for the majority-dummy baseline (ICBHI Score rewards
specificity on Normal cheaply): CNN predicts "normal" correctly far more
often (Sp=0.608 vs ~0.36 for E01/E02) but at the cost of almost all abnormal
sensitivity, especially `both`. The ICBHI Score ranking must not be read as
"CNN is the better model" — macro-F1 and per-class recall are the
trustworthy signal here, consistent with docs/03's warning that ICBHI Score
is supplementary, never standalone.

## Conclusion for D28

Per docs/04 ("CNN loses to SVM: ship SVM as recommended model, retain CNN
comparison... engineering rigor, not forced deep-learning supremacy") and
D21's standing choice of logistic regression (E01) as the classical-ML
candidate (statistically indistinguishable from E02, no selection bias): the
classical ML candidate remains the stronger, honest choice for model freeze.
CNN is retained as a documented research comparison, not shipped as primary.

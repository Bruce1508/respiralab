# Classical ML baseline review — D21 (Gate G3)

Primary split (D07, seed=42): train=84 patients/4649 cycles, val=21 patients/1108
cycles, test=21 patients/1141 cycles (test untouched below — `test_locked=true`
per docs/03). Split hash `38e11a7f...` (manifests/splits_primary.csv), identical
across every run cited here.

## Candidates evaluated

| Experiment | Model | macro-F1 (val) | 95% patient-bootstrap CI | ICBHI Score (val) |
|---|---|---|---|---|
| E00 | majority dummy | 0.1800 | — (D20 only ran E01/E02) | 0.500 |
| E00 | stratified dummy | 0.2549 | — | 0.367 |
| E01 | logistic regression | 0.3782 | [0.289, 0.439] | 0.466 |
| E02 | RBF SVM (C=1.0, gamma=0.01, best of 9-config grid) | 0.3872 | [0.280, 0.460] | 0.476 |

Per-class recall (val, support in parentheses):

| Model | normal (623) | crackle (291) | wheeze (126) | both (68) |
|---|---|---|---|---|
| logistic regression | 0.358 | 0.698 | 0.571 | 0.044 |
| RBF SVM (best) | 0.369 | 0.687 | 0.579 | 0.147 |

ICBHI Score (Sp/Se) computed by re-running D19's `icbhi_metric.icbhi_score()`
against each model's real val predictions (ad hoc, not a committed script —
same pattern as the D08-D14 real-data verification runs); all other numbers
come directly from the already-committed `results/E00|E01|E02/dev_scores.json`
and `results/seed_comparison.csv`.

## Leakage and protocol checks

- **Patient-disjoint split**: zero patient overlap across train/val/test,
  asserted in code and independently re-verified (D07, commits `eda60d2`/`c0a1bed`,
  Gate G1 PASS).
- **Feature computation blind to split**: identical audio produces an identical
  `feature_hash` regardless of assigned split label, directly unit-tested
  (D13, commits `91a589e`/`9bf0e9c`).
- **Scaler fit train-only**: both E01 and E02 use an `sklearn.Pipeline`, whose
  `.fit()` only ever fits the `StandardScaler` on the data passed to it;
  `test_scaler_fits_only_on_train_not_dev` (D16) directly asserts the fitted
  scaler's mean matches the train-only mean, not train+dev combined, even when
  dev is deliberately shifted far outside train's range.
- **Model/hyperparameter selection on val only**: E02's 9-trial grid search
  (D17) only ever calls `load_split_features(..., "train", ...)` and
  `(..., "val", ...)` — `test` is never loaded by `svm_search.py` at all.
- **Train-seed reproducibility**: both E01 and E02 are bit-identical macro-F1
  across train seeds 13/21/42 (D20) — expected, since `lbfgs` and
  `SVC(probability=False)` don't consume `random_state`; confirms there is no
  hidden training randomness masking the comparison above.

## Honest comparison

- E01 vs E02's macro-F1 gap (0.3782 vs 0.3872, 0.009) is small relative to
  either model's patient-bootstrap CI half-width (~0.075-0.09, D20) — on a
  21-patient val set, this gap is **not statistically distinguishable**.
- E02's reported score additionally carries a mild selection bias: it is the
  best of 9 (C, gamma) configurations all evaluated on the same 21-patient val
  set used to report the final number — a "winner's curse" effect. E01 has no
  such search. This makes the already-small E02 > E01 gap even less trustworthy
  than its magnitude alone suggests.
- ICBHI Score agrees in direction (SVM 0.476 > logreg 0.466) but the metric has
  a known blind spot, visible right here in the baselines: the **majority**
  dummy scores 0.500 by trivially predicting Normal for every cycle (Sp=1.0,
  Se=0.0, mean=0.5), *higher* than the **stratified** dummy's 0.367 despite
  stratified actually attempting abnormal predictions. This is exactly why
  docs/03 mandates macro-F1 as the primary metric and ICBHI Score only as a
  supplement — never read alone.
- Both real models share the same weak point: the `both` class (support=68,
  the rarest) has by far the lowest recall of the four (0.044 logreg / 0.147
  SVM) — a dataset-imbalance finding already flagged in D16/D17/D20, restated
  here as a carried-forward limitation rather than newly discovered.

## Decision

**Classical ML candidate: logistic regression (E01).**

1. Statistically indistinguishable from E02 given bootstrap CI overlap.
2. No val-set selection bias (E02's point estimate reflects the best of 9
   trials scored on the same val set reported above).
3. Simpler and fully interpretable (linear coefficients), with no
   hyperparameter search budget to maintain going forward.

RBF SVM (E02) is retained as a secondary reference model, not discarded — its
marginally higher point estimates are noted but not treated as a proven
improvement given the overlap above.

## Gate G3 checklist

- At least 2 baselines compared: **yes** — 4 total (majority dummy, stratified
  dummy, logistic regression, RBF SVM).
- Honest comparisons: **yes** — macro-F1 and ICBHI Score reported together
  with bootstrap CIs; E02's selection bias explicitly flagged rather than
  hidden; ICBHI Score's own blind spot (majority dummy scoring 0.500) called
  out rather than cherry-picked in its favor.
- No leakage: **yes** — see "Leakage and protocol checks" above, each claim
  backed by a specific prior commit/test rather than restated from memory.

**Gate G3 decision: PASS.** Deep Learning phase (D22+) may proceed.

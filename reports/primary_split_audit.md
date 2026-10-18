# Primary patient-disjoint split audit (D07, Gate G1)

Seed: 42. Patient-level StratifiedKFold on each patient's dominant
cycle label (test: 6 folds, val: 5 folds of the
remainder). Ratios are guidance (~70/15/15), not an exact target.

## Leakage checks
Zero patient overlap across train/val/test, zero recording overlap between
train and test (asserted in `tests/test_splits.py`, re-verified here: all
126 patients land in exactly one split).

## Rare-class coverage
All 4 classes present with at least 1 cycle in every split.

## Per-split counts
| split | patients | recordings | cycles | normal | crackle | wheeze | both |
|---|---|---|---|---|---|---|---|
| train | 84 | 630 | 4649 | 2293 | 1337 | 654 | 365 |
| val | 21 | 156 | 1108 | 623 | 291 | 126 | 68 |
| test | 21 | 134 | 1141 | 726 | 236 | 106 | 73 |

# D44 — 5-fold patient-grouped CV over train+val (E01 reference)

**Protocol:** `5-fold patient-grouped stratified CV over train+val (ADR-0001 D3); test excluded`
(`results/E01_cv/cv_scores.json`). Split hash `38e11a7fa958b408af835e4dd89ba31364482b7734042606f127ef92900068da`
(unchanged, D07). Pool: 105 train+val patients, 5,757 cycles. The 21 primary-test patients never
enter any fold (asserted in `assert_no_patient_crosses_folds`, re-checked against the real manifest).

## Why this exists

v1's single val split (21 patients) and the one-time test split (21 patients) disagreed sharply
per class: wheeze recall 0.571 (val) -> 0.104 (test); both recall 0.044 (val) -> 0.329 (test)
(`results/E01/dev_scores.json`, `reports/final_test_results.json`). That is consistent with high
variance from small patient-level holdouts, not necessarily a real shift. ADR-0001 therefore
requires v2 model selection to use CV mean +/- sd over the pooled train+val patients instead of a
single split.

## E01 under the new protocol (v2 reference point; not comparable to v1 single-split numbers)

| metric | mean | sd (across 5 folds) |
|---|---|---|
| macro-F1 | 0.3654 | 0.0369 |
| normal recall | 0.4433 | 0.0778 |
| crackle recall | 0.4933 | 0.1392 |
| wheeze recall | 0.3285 | 0.0920 |
| both recall | 0.3017 | 0.0622 |

Per-fold patient/cycle counts and per-fold macro-F1 (`results/E01_cv/cv_scores.json`):

| fold | train patients | val patients | val cycles | macro-F1 |
|---|---|---|---|---|
| 0 | 85 | 20 | 1153 | 0.3137 |
| 1 | 85 | 20 | 1143 | 0.3453 |
| 2 | 82 | 23 | 1144 | 0.3685 |
| 3 | 86 | 19 | 1197 | 0.4001 |
| 4 | 82 | 23 | 1120 | 0.3992 |

## Reading this

- Every class's CV-mean recall now lands in a narrower, more plausible band (0.30-0.49) than either
  single split alone showed. This is the expected effect of averaging over 5 independent
  21-23-patient holdouts instead of reporting one.
- `both`-recall mean 0.302 sits between v1's val (0.044) and test (0.329) extremes - neither prior
  single-split number should be read as "the" both-class recall for E01.
- This CV run is the **v2 reference baseline**, not a new v1 result. v1's frozen numbers
  (`reports/model_freeze.md`, `reports/final_test_results.json`) are unchanged and still the
  published ones.
- E05 (frozen-encoder probe, D48) is compared against this table's macro-F1 mean, per ADR-0001 D4b's
  go/stop rule for E06.

## Verification

- `uv run python -m respiralab.models.traditional ...` reproduced `results/E01/dev_scores.json`
  bit-for-bit before this run (v1 untouched).
- `uv run pytest -q` -> 175 passed (170 + 5 new `tests/evaluation/test_group_cv.py`).
- `uv run ruff check src tests` -> All checks passed.

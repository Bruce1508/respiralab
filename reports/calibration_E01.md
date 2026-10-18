# D45 — Temperature scaling and ECE for E01

**Protocol:** `temperature scaled on out-of-fold CV predictions (ADR-0001 D5); never on test`
(`results/E01_calibration/calibration_scores.json`). Split hash `38e11a7fa958b408af835e4dd89ba31364482b7734042606f127ef92900068da`
(unchanged, D07). Same 105-patient, 5,757-cycle train+val pool and 5-fold grouping as D44
(`src/respiralab/evaluation/group_cv.py`). Temperature is fit by MLE on out-of-fold logits only —
every prediction comes from a fold that did not train on that sample — and is never touched by the
primary test.

## Result

| metric | value |
|---|---|
| fitted temperature | **2.0069** |
| ECE before scaling (T=1) | 0.0950 |
| ECE after scaling | **0.0426** |

E01's raw `decision_function` logits are overconfident (fitted T > 1): a predicted-probability bin
around 0.9 confidence was only ~54-64% accurate in several bins before scaling. After dividing
logits by T=2.0069, 10-bin ECE drops by more than half (0.095 -> 0.043). Reliability diagram:
`reports/figures/calibration_E01.png` (before/after, confidence vs. accuracy, 10 equal-width bins).

## Reading this

- This calibrates E01's probability outputs, not its decision boundary — `argmax` predictions and
  macro-F1 (D44, `reports/v2_group_cv_E01.md`) are unaffected by temperature scaling.
- A calibrated probability matters for the Streamlit demo's confidence display and for any future
  OOD/confidence-threshold gate (ADR-0001 D7).
- Still not comparable to v1's published numbers — this is v2-protocol scaffolding, run on the same
  CV folds as D44.

## Verification

- `uv run pytest -q` -> 182 passed (175 prior + 7 new in `tests/evaluation/test_calibration.py`, all
  on synthetic data — including a test that recovers a known injected temperature within 15%).
- `uv run ruff check src tests` -> All checks passed.
- `uv run python -m respiralab.evaluation.calibration ...` -> wrote
  `results/E01_calibration/calibration_scores.json` and `reports/figures/calibration_E01.png`.

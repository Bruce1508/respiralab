# D33 — Single-factor ablation: `class_weight` (dev data only, test untouched)

**Variable changed (only one):** `LogisticRegression(class_weight=...)` —
`None` vs `"balanced"` (E01's frozen setting). Scaler, split, seed (42),
solver, and every other hyperparameter are identical to E01. Evaluated on
`val` (dev) only — the primary test set stays closed per D29/Gate G4.

## Results

| class_weight | macro_f1 (val) | 95% patient-bootstrap CI | both_recall (val) |
|---|---:|---|---:|
| `None` | 0.3938 | [0.2995, 0.4493] | 0.0000 |
| `"balanced"` (frozen E01) | 0.3782 | [0.2892, 0.4393] | 0.0441 |

## Performance uncertainty

The two macro-F1 values differ by only 0.0156, and their 95% patient-
bootstrap CIs overlap almost entirely (`[0.2995, 0.4493]` vs
`[0.2892, 0.4393]`) — **this macro-F1 difference is not distinguishable
from noise** given only 21 val patients, consistent with D21's finding that
small macro-F1 gaps at this phase routinely fall inside the bootstrap CI.

What *is* a clear, non-noise effect: unweighted training completely zeroes
`both`-class recall (0.0 vs 0.044) on this val set. Macro-F1 alone would not
have flagged this — a reminder (consistent with D21/D27's ICBHI Score
blind-spot findings) that a single aggregate metric can mask a real
minority-class failure. This does not change the D28 freeze decision
(E01 with `class_weight="balanced"` remains frozen, and the primary test
result at D29 is already final and not reopened here) — it is reported as
a documented, test-free sensitivity check, per this task's acceptance
criterion.

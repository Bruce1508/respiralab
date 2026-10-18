# D48 — E05: frozen AST embeddings -> logistic regression (grouped CV)

**Protocol:** `E05: frozen AST embeddings -> logistic regression, grouped CV (ADR-0001 D4b)`
(`results/E05/probe_scores.json`). Same split hash, same 105-patient/5,757-cycle pool, same 5 folds
as D44's E01 reference (`src/respiralab/evaluation/group_cv.py`, seed 42). Embeddings: 768-d pooled
output of `MIT/ast-finetuned-audioset-10-10-0.4593` (D47), computed on cycle audio run through
`dsp.encoder_input.prepare_encoder_input` (16 kHz, 1800 Hz common-band cutoff, zero-pad to 6.0 s).
Head: the same `StandardScaler + LogisticRegression(class_weight="balanced")` pipeline as E01
(`models/traditional.py::build_pipeline`) — only the input features differ.

## Result

| metric | E01 (D44 reference) | E05 (this probe) |
|---|---|---|
| macro-F1 mean | 0.3654 | **0.4448** |
| macro-F1 sd | 0.0369 | 0.0226 |
| normal recall | 0.4433 | 0.6142 |
| crackle recall | 0.4933 | 0.5178 |
| wheeze recall | 0.3285 | 0.3916 |
| both recall | 0.3017 | 0.2939 |

**E06 go/stop decision (ADR-0001 D4b rule: E05 > E01 mean + 1 fold-sd):**
threshold = 0.3654 + 0.0369 = **0.4022**. E05's 0.4448 clears it by 0.0426 (more than a full
additional sd). **Decision: GO** — D49 (bounded fine-tune, 3 seeds) is authorized to proceed.

## Reading this

- AST embeddings beat E01's 46 handcrafted features on macro-F1 and on 3 of 4 per-class recalls
  (normal, crackle, wheeze), with a *tighter* fold spread (sd 0.0226 vs 0.0369) — more stable
  across patient groups, not just higher on average.
- `both`-class recall is essentially flat (0.294 vs E01's 0.302) — frozen AudioSet features alone
  don't fix the minority-class problem the ADR exists to address. That's D50's job (E07:
  balanced-sampling / focal-loss / mixup ablations), not this probe's.
- Still not comparable to v1's published single-split numbers (`reports/model_freeze.md`) — this is
  the v2-protocol table.

## Verification

- `uv run pytest -q` -> 198 passed (195 prior + 3 new in `tests/models/test_encoder_probe.py`, all
  synthetic/monkeypatched — no real model load in the unit tests).
- `uv run ruff check src tests` -> All checks passed.
- Smoke-tested embedding extraction on 6 real cycles (correct shape, finite values) before launching
  the full 5,757-cycle run in the background (~95 CPU-minutes on this machine).
- Embeddings cached per-cycle under `artifacts/embeddings_cache/` (gitignored, resumable — confirmed
  `assert_no_patient_crosses_folds` passes on the real manifest, same as D44).

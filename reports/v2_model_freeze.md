# D52 — v2 model freeze and pre-test protocol declaration (Gate G7)

**Frozen at commit:** `79ab2d8` (D52 freeze script), evidence file written by this run of
`src/respiralab/models/v2_freeze.py` at `results/v2_frozen/v2_freeze.json`.
**Frozen at date:** 2026-10-20
**Primary split hash (frozen, unchanged since D07/v1):** `38e11a7fa958b408af835e4dd89ba31364482b7734042606f127ef92900068da`
— matches `results/E01_cv/cv_scores.json` and v1's `reports/model_freeze.md` exactly. v1 hashes reproduce.
**Encoder checkpoint:** `MIT/ast-finetuned-audioset-10-10-0.4593`, sha256 `ae0c1e2a...69924d`.

## Declared frozen model

**E06 — bounded AST fine-tune** (`src/respiralab/models/finetune.py`):
- Architecture: pretrained AST encoder, last **2 of 12** transformer blocks + final layernorm
  + a new `Linear(768, 4)` head unfrozen; rest of the encoder frozen.
- Optimizer: Adam, lr=1e-5, batch size 8, plain cross-entropy loss.
- Early stopping: max 3 epochs, patience 1 on val macro-F1 (per-fold best-epoch rollback via
  `track_best_state=True`).
- Final model seed: 42. Final training: retrain on **100% of train+val** (5,757 cycles) for
  `FINAL_EPOCHS=2` — 2 is the median best-epoch count (`best_idx+1`) across the 15 D49 grid
  runs (seven 1s, four 2s, four 3s); no val split remains to early-stop against once training
  on the full pool.
- Decision rule: hard 4-class argmax; no separate binary threshold is tuned (task is 4-class).

## Grouped-CV evidence for this choice

All numbers below are 5-fold patient-grouped CV over train+val (105 patients, 5,757 cycles,
test patients excluded throughout), per ADR-0001.

| Model | Macro-F1 (mean ± std) | Source |
|---|---|---|
| **E01 CV reference** (logistic regression, v1 baseline) | **0.3654 ± 0.0330** | `results/E01_cv/cv_scores.json` |
| E05 (frozen AST embeddings, linear head) | 0.4448 ± 0.0226 | `results/E05/...` (D44/D45) |
| **E06 (bounded fine-tune, 3-seed × 5-fold grid)** | **0.4791 ± 0.0401** | `results/E06/grid_scores.json`, commit `571d782` |

E06 beats E05 in 14 of 15 seed×fold combinations (only miss: seed 13 fold 1, 0.4159 vs 0.4196)
and clears the E05 mean+1sd bar. This is the basis for choosing E06 over E05 and over the E01
baseline (D49, decided with the user — not re-litigated here).

Supporting single-factor ablations (not fused into v2.0, see "Rejected / deferred" below):
- **D50 E07** — balanced sampling raises `both`-class recall to 0.3683 ± 0.1067 vs 0.2634 ± 0.1516
  for class-weighting (`results/E07/ablation_scores.json`, `reports/v2_ablation_both.md`).
- **D51 E08** — zero-padding (0.4448 macro-F1 / 0.2939 both-recall) beats repeat-padding
  (0.4243 / 0.2296) on the frozen-embedding probe (`results/E08/probe_scores.json`,
  `reports/v2_padding_ablation.md`).

## Calibration (fit off-test)

Out-of-fold temperature scaling, fit on val logits pooled across the same 5 grouped-CV folds
(seed 42) used for the grid — **no primary test cycle is touched**:

| | Value |
|---|---|
| OOF cycles | 5,757 |
| Temperature | 1.3288 |
| ECE before scaling | 0.0819 |
| ECE after scaling | 0.0112 |

Per-fold best val macro-F1 during the OOF calibration runs (0.4592 / 0.4344 / 0.4816 / 0.5171 /
0.4727) matches the seed-42 row of the D49 grid exactly — confirms this run is a faithful
reproduction of the registered protocol, not a new experiment.

**Known gap:** per-class (incl. `both`) recall for the final E06 configuration was not recorded
by `finetune.py`'s grid (macro-F1 only) and is not cheap to recompute without re-running the CV
folds. Not fabricated here. Directional evidence from E07's ablation (`both`-recall 0.26–0.37
across sampling strategies on an unrelated linear-head configuration) should **not** be read as
E06's own `both`-recall. Recommended v2.1 follow-up: instrument `run_fold` to also return
per-class recall and recompute from the already-cached OOF logits/labels at low cost (mostly a
code change, not a retrain) — see "not yet tried" below.

## Final weights

| | Value |
|---|---|
| Path | `results/v2_frozen/v2_model_state_dict.pt` |
| sha256 | `e3ddc62d5941f7839f808d0ebcf5b3f5d8c175c6d368478ba32a812648c037ef` |
| Trained on | 5,757 cycles (100% train+val) |
| Final-train history | epoch 0: train macro-F1 0.4882, loss 0.8848; epoch 1: train macro-F1 0.6132, loss 0.7233 |

Train-set macro-F1 rising to 0.61 after 2 epochs on the full pool (vs ~0.46–0.52 val macro-F1
during the fixed-epoch CV folds) is expected: this is in-sample fit with no held-out early
stopping, not a claim about generalization. The frozen generalization estimate remains the
grouped-CV numbers above; D53 will provide the one-time primary-test number.

## Rejected / deferred (not fused into v2.0)

- **balanced_sampling (E07)** — kept out to preserve ADR-0001's single-factor discipline;
  documented as v2.1 work, not combined with E06 without a new pre-registered experiment.
- **MLX / LoRA / QLoRA** (user-researched alternatives) — rejected for v2.0: mlx-lm has no AST
  support; LoRA doesn't address the actual bottleneck (compute per step, not memory — AST is
  only ~350MB); QLoRA targets multi-billion-parameter LLMs and would cost accuracy here;
  bitsandbytes has poor MPS support. Switching methods now would also invalidate D49's evidence
  and require a new ADR.
- **Caching frozen-prefix activations** to speed up fine-tuning (~5-6x) — mathematically
  identical, candidate for v2.1, not needed for the already-frozen v2.0.

## Gate G7 go/no-go checklist (`planning/QUALITY_GATES.md`)

1. **Grouped-CV evidence for every v2 choice?** Yes — E06 vs E05 vs E01 table above, each citing
   a committed JSON artifact and commit hash (D49/D50/D51).
2. **E01 CV reference present?** Yes — 0.3654 ± 0.0330, recomputed directly from
   `results/E01_cv/cv_scores.json` in this session, same grouped-CV protocol as E06.
3. **Calibration fit off-test?** Yes — temperature fit on pooled OOF val logits from the 5
   grouped-CV folds; primary test split is never loaded by `v2_freeze.py` (only
   `--splits-csv` is hashed, not read for labels).
4. **v1 hashes reproduce?** Yes — split hash `38e11a7f...` is byte-identical between
   `results/E01_cv/cv_scores.json`, v1's `reports/model_freeze.md`, and this run's
   `v2_freeze.json`.
5. **Weights hash recorded (ADR-0001)?** Yes — `e3ddc62d...` above, computed by
   `sha256_of_file` in `v2_freeze.py`, alongside the encoder checkpoint's own hash.

**Gate G7 decision: PASS.** v2 model (E06, bounded fine-tune, final weights
`e3ddc62d5941f7839f808d0ebcf5b3f5d8c175c6d368478ba32a812648c037ef`) is frozen. D53 (one-time
primary-test evaluation) may proceed **only after explicit user confirmation** — this report
does not itself authorize opening the test set.

**No primary-test numbers are cited anywhere in this report.**

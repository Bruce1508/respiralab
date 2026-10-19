# D50 — E07a/b/c: single-factor `both`-class ablations

Protocol: ADR-0001 D4b. Same D44 grouped-CV folds (`fold_split_seed=42`), same
D48 frozen AST embeddings (no new feature extraction). A tiny linear head
(`nn.Linear(768, 4)`) is trained per fold with early stopping on val macro-F1
(patience=5, max 50 epochs). Each ablation changes exactly one factor
relative to the `reference` run (class-weighted cross-entropy, matching
E01/E05's `class_weight="balanced"`).

## Results (5-fold CV mean ± sd)

| Method | macro-F1 | `both`-recall |
|---|---|---|
| reference (class-weighted) | 0.4587 ± 0.0524 | 0.2634 ± 0.1516 |
| **balanced_sampling** | **0.4778 ± 0.0457** | **0.3683 ± 0.1067** |
| focal_loss (γ=2) | 0.4799 ± 0.0516 | 0.2930 ± 0.1453 |
| mixup (α=0.4, train-fold-only) | 0.4747 ± 0.0305 | 0.2849 ± 0.2313 |

## Reading

- **balanced_sampling** is the clear winner for the `both` class: +0.105
  recall over the reference, with the lowest variance of the three
  ablations and a macro-F1 gain too. This is the one worth carrying into
  D52 if the minority class matters for the final v2 model.
- **focal_loss** nudges macro-F1 up slightly but barely moves `both`-recall
  — on this data, reweighting the loss by class frequency (already done by
  the reference) captures most of what focal loss offers; the extra
  (1-pt)^γ down-weighting of easy examples doesn't add much here.
- **mixup** has the highest variance by far (sd 0.23 on `both`-recall, out of
  a ~0.28 mean) — unstable on this minority class with this few samples per
  fold. Not recommended as-is.

All three ablations beat the reference on macro-F1; only balanced_sampling
clearly beats it on the metric D50 was actually about (`both`-recall).

Raw scores: `results/E07/ablation_scores.json`.

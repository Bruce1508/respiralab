# D51 — E08: repeat-pad vs zero-pad

Protocol: ADR-0001 D4b. Single factor changed from the E05 frozen-embedding
probe (D48): `pad_mode` in `dsp/encoder_input.prepare_encoder_input`. Same
cycles, same D44 folds, same AST checkpoint, same logistic-regression head
(`class_weight="balanced"`) -- only the padding of short cycles to 6.0s
differs.

## Results (5-fold CV mean ± sd)

| Pad mode | macro-F1 | `both`-recall |
|---|---|---|
| **zero** (E05 reference) | **0.4448 ± 0.0226** | **0.2939 ± 0.0757** |
| repeat (E08) | 0.4243 ± 0.0298 | 0.2296 ± 0.0592 |

## Reading

Zero-pad wins on both macro-F1 (-0.0205 for repeat) and `both`-recall
(-0.0643 for repeat). Repeating a short cycle's audio to fill 6.0s
apparently confuses the pretrained AST encoder more than silence does --
plausible, since repeated breathing-cycle content isn't something the
AudioSet-pretrained checkpoint saw in training, while silence is a more
"neutral" padding signal it can effectively ignore. v1's original choice
(zero-pad, `models/dataset.py::pad_or_crop`) is confirmed correct for v2 too;
no amendment needed.

Raw scores: `results/E08/probe_scores.json` (vs `results/E05/probe_scores.json`
for the zero-pad reference).

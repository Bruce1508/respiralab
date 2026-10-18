# RespiraLab — Final technical report

**Research/educational demonstration. Not a diagnostic device.** Sound-event
labels (Normal/Crackle/Wheeze/Both), not disease diagnoses. Full detail and
citations for every number below live in the per-day `reports/*.md` and
`planning/PROGRESS_LOG.md` files this report summarizes — nothing here is
restated from memory.

## Dataset and split

ICBHI 2017: 920 recordings, 126 patients, 6,898 cycles (D03/D04, exact
match to official totals). Primary split (D07): patient-disjoint,
train=84p/4649c, val=21p/1108c, test=21p/1141c, `split_hash=38e11a7f...`,
zero patient overlap (independently re-verified). The official ICBHI
challenge split is a **separate, non-disjoint protocol** — patients 156
and 218 appear in both its train and test (D06, re-confirmed D30) —
reported separately, never combined with the primary split's numbers.

![Label distribution](figures/labels.png)
*Figure 1 — cycle count per class across the full dataset (D05): heavily
imbalanced toward Normal, with `both` the rarest class.*

![Device distribution](figures/devices.png)
*Figure 2 — recording device distribution (D05): 4 heterogeneous
acquisition devices, a known confound discussed in the model card.*

## Pipeline (DSP, Gate G2 PASS at D14)

Deterministic cycle extraction → Nyquist-safe resampling → optional
filters → handcrafted features (MFCC/spectral/46-dim, D12) and log-Mel
tensors (D22, for the CNN path). 19.2% of recordings have measurable
clipping (D14) — a known data-quality limitation, not corrected.

![Filter before/after](figures/filter_before_after.png)
*Figure 3 — bandpass filter effect on a real crackle cycle (D11):
in-band energy retained 0.886, out-of-band near zero.*

## Models (Gate G3 PASS at D21, Gate G4 PASS at D28)

Four candidates compared honestly: majority/stratified dummy (E00, D15),
logistic regression (E01, D16), RBF SVM (E02, D17), compact CNN (E03,
D23-D26). **E01 was frozen** (`reports/model_freeze.md`): E02's small
macro-F1 edge was inside both models' patient-bootstrap CI and carried
val-set selection bias (D21); the CNN underperformed on every metric and
failed completely on `both` (recall=0.0, D27).

## Results

| protocol | macro_f1 | icbhi_score | source |
|---|---:|---:|---|
| val (dev) | 0.3782 | 0.4656 | `results/E01/dev_scores.json` |
| **primary test (one-time, D29)** | **0.2861** | **0.3762** | `reports/final_test_results.json` |
| official split (separate, non-disjoint, D30) | 0.3366 | — | `reports/official_protocol_comparison.md` |

The test macro-F1 is honestly lower than val — a real generalization gap,
not adjusted after the fact. The primary test was opened exactly once
(D29) and is now closed; no subsequent tuning occurred.

## Robustness and error analysis

- **Noise (D31):** macro-F1 degrades at every tested SNR vs clean (worst
  -0.0934 at 0dB). Supported-use claim narrowed to ICBHI-comparable
  recording conditions.
- **Errors (D32):** of 144 misclassified wheeze/both test cycles, errors
  cluster into two non-random types — missed-abnormal-to-Normal, and
  confusion between adjacent abnormal classes.
- **Ablation (D33):** removing `class_weight="balanced"` barely moves
  macro-F1 (within the bootstrap CI) but zeroes `both`-class recall —
  a real effect the headline metric alone would hide.

## Reproducibility

Every result above cites a commit (full history in `git log` on
https://github.com/Bruce1508/respiralab), a fixed seed, and the exact
split/preprocess hash. `uv sync && uv run pytest -q` reproduces 170
passing tests from a fresh clone with no ICBHI data present (D38/D39
synthetic smoke test + CI). Real-data numbers require the user's own
ICBHI download (`data/README.md`) — raw audio is never committed.

## Limitations (full list: `reports/model_card.md`)

`both`-class recall weak across every model tried; 19.2% recording
clipping; noise sensitivity; ICBHI Score's Normal-specificity blind spot;
dataset scope limited to 126 patients / 4 devices. No diagnosis claim is
made anywhere in this pipeline.

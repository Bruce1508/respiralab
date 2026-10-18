# RespiraLab v1.0 — release notes

## Scope

A reproducible, leakage-checked respiratory sound-event classifier
(Normal/Crackle/Wheeze/Both) on ICBHI 2017. Research/educational tool,
**not a diagnostic device** — covers Foundation through Launch (D01-D42)
of the project's execution plan.

## What's included

- Patient-disjoint primary split (zero overlap, independently verified
  twice), plus a disclosed, separately-reported official-split comparison.
- Deterministic DSP pipeline: resampling, filtering, handcrafted features,
  log-Mel tensors (Gate G2 PASS).
- 4 compared candidates: dummy baselines, logistic regression (frozen),
  RBF SVM, compact CNN (Gate G3 PASS).
- Frozen model (Gate G4 PASS) + one-time primary test evaluation
  (macro_f1=0.2861), official-split comparison, noise robustness check,
  error analysis, single-factor ablation, and a model card with 7 explicit
  limitations (Gate G5 PASS).
- Offline inference with input validation, a Streamlit demo UI, synthetic
  CI (no real data needed), and a latency/memory benchmark
  (p50=2.43ms, p95=3.54ms).
- This release notes file + `demo_script.md` + an independently
  re-verified fresh-clone walkthrough (Gate G6 PASS).

## Known limitations (full detail: `reports/model_card.md`)

`both`-class recall is weak across every model tried (worst: CNN at
0.0); 19.2% of recordings have measurable clipping; macro-F1 degrades
under noise; ICBHI Score has a Normal-specificity blind spot; results
are scoped to 126 patients and 4 acquisition devices.

## Verification

- `uv sync && uv run pytest -q` → `169 passed, 1 skipped` on a genuinely
  fresh clone (no ICBHI data present), re-verified for this release.
- CI (`.github/workflows/ci.yml`): ruff + import smoke + CLI smoke +
  full test suite, every push/PR.
- No secrets, no files >1MB, no raw audio, no model pickles committed
  (`release_checklist.md`).

## Not included (explicit non-goals, per the PRD)

Clinical diagnosis of any kind, medical device compliance, real-time
hardware integration, state-of-the-art performance claims.

# Build plan — D23 through D42

Mirrors `../../RespiraLab_Project_Pack/planning/tasks.json` (source of truth;
do not edit task definitions here, only check boxes and log notes). Dependency
order = task order, same as tasks.json. One commit per task, per the build
skill's default loop (RED → GREEN → regression → ruff → commit).

## Phase: Deep Learning (D23–D28)
- [x] **D23** — Train minimal compact CNN over log-Mel, fixed config. Deliverable: `cnn.py`, `results/E03`. Acceptance: loss decreases on tiny overfit smoke batch; checkpoint saved. (commits `425236d`, `a1ce5bf`)
- [x] **D24** — Validation logging, early stopping, checkpoints, resume. Deliverable: `training.py`, `train_cnn.yaml`. Acceptance: run can resume; best chosen by validation only. (commits `35c52fa`, `d09584c`)
- [x] **D25** — Training-only augmentations; test inherited patient IDs. Deliverable: `augment.py`, tests. Acceptance: no augmented child assigned to val/test. (commit `03a1712`)
- [x] **D26** — Controlled CNN training over 3 seeds; capture compute cost. Deliverable: `results/E03_seed_runs`. Acceptance: each run records config/commit/hardware/runtime. (commit `cc09a05`)
- [x] **D27** — Compare CNN vs best classical ML, identical validation protocol. Deliverable: `reports/model_comparison.md`. Acceptance: same test-free dev set; minority recall reported. (committed)
- [x] **D28** — Freeze model selection, write pre-test protocol declaration. Deliverable: `reports/model_freeze.md`. Acceptance: **Gate G4** — config/threshold/metrics/hashes frozen. (commit `5406c76`, GATE G4: PASS)

> Gate G4 PASS logged in PROGRESS_LOG.md (2026-10-18), all 5 go/no-go questions answered with citations. User authorized continuing without a live pause at each remaining gate, provided criteria are genuinely met. Proceeding to D29.

## Phase: Validation (D29–D34)
- [x] **D29** — Run FINAL untouched primary test evaluation, report once. Deliverable: `reports/final_test_results.json`. Acceptance: no subsequent test-driven tuning; class counts included. **Irreversible — test is opened exactly once here.** (commit `ce91dbb`, macro_f1_test=0.2861, test now closed — no further tuning permitted)
- [x] **D30** — Official-split comparative experiment, separate protocol. Deliverable: `reports/official_protocol_comparison.md`. (committed)
- [x] **D31** — Fixed-noise/SNR robustness checks, no model changes. Deliverable: `reports/robustness.md`. (committed)
- [ ] **D32** — Minority-class failure inspection by device/site/error type. Deliverable: `reports/error_analysis.md`.
- [ ] **D33** — Single-factor ablation on dev data only. Deliverable: `reports/ablation.md`.
- [ ] **D34** — Model card + clinical/recording caveats. Deliverable: `reports/model_card.md`. Acceptance: **Gate G5** — no diagnosis claims, limitations explicit.

> **STOP — Gate G5 human sign-off required before Product phase.**

## Phase: Product (D35–D40)
- [ ] **D35** — Validated offline predictor + WAV input guards. `inference/predict.py`.
- [ ] **D36** — Streamlit upload + waveform + spectrogram UI. `app/streamlit_app.py`.
- [ ] **D37** — Probabilities, method label, provenance in UI.
- [ ] **D38** — Synthetic integration fixtures + E2E UI smoke. `tests/integration/`.
- [ ] **D39** — CI lint/test/import checks + sensitive-data scan. `ci.yml`.
- [ ] **D40** — Inference latency/memory benchmark + packaged release. `reports/performance.md`.

## Phase: Launch (D41–D42)
- [ ] **D41** — Honest technical report, citations, figure captions. `reports/final_report.md`.
- [ ] **D42** — Fresh-clone walkthrough, demo, publish v1.0. Acceptance: **Gate G6**.

> **STOP — Gate G6 + explicit user go-ahead required before anything is published/pushed.**

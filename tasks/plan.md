# Build plan — D23 through D57

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
- [x] **D32** — Minority-class failure inspection by device/site/error type. Deliverable: `reports/error_analysis.md`. (committed)
- [x] **D33** — Single-factor ablation on dev data only. Deliverable: `reports/ablation.md`. (committed)
- [x] **D34** — Model card + clinical/recording caveats. Deliverable: `reports/model_card.md`. Acceptance: **Gate G5** — no diagnosis claims, limitations explicit. (commit `d526971`, GATE G5: PASS)

> Gate G5 PASS logged in PROGRESS_LOG.md. Proceeding to Product phase per user's standing authorization to continue without a live pause at each gate.

## Phase: Product (D35–D40)
- [x] **D35** — Validated offline predictor + WAV input guards. `inference/predict.py`. (commit `7d2011f`)
- [x] **D36** — Streamlit upload + waveform + spectrogram UI. `app/streamlit_app.py`. (commit `5ad0e02`)
- [x] **D37** — Probabilities, method label, provenance in UI. (commit `5ad0e02`; screenshot sub-deliverable not produced - no browser access this session)
- [x] **D38** — Synthetic integration fixtures + E2E UI smoke. `tests/integration/`. (commit `b4904aa`)
- [x] **D39** — CI lint/test/import checks + sensitive-data scan. `ci.yml`. (commit `994a010`; also fixed a real CI-breaking test gap)
- [x] **D40** — Inference latency/memory benchmark + packaged release. `reports/performance.md`. (commit `b522e2f`)

## Phase: Launch (D41–D42)
- [x] **D41** — Honest technical report, citations, figure captions. `reports/final_report.md`. (commit `de8eb4c`)
- [x] **D42** — Fresh-clone walkthrough, demo, publish v1.0. Acceptance: **Gate G6**. (commit `4660fc0`, GATE G6: PASS - genuinely re-cloned and re-verified, not assumed)

> **D01-D42 ALL COMPLETE. All 6 gates (G1-G6) PASS.** Pushed to origin and public (https://github.com/Bruce1508/respiralab); demo live on Streamlit Community Cloud (https://respiralab-6crxyqyputfxwbmuatdsf9.streamlit.app). Not git-tagged.

## Phase: V2 (D43–D57) — per `docs/adr/0001-v2-pretrained-encoder-and-evaluation.md`
- [x] **D43** — Write ADR-0001 and pre-register the v2 roadmap (D44-D57). Deliverable: `docs/adr/0001-v2-pretrained-encoder-and-evaluation.md, tasks D44-D57`. Acceptance: ADR covers sample rate, selection protocol, test-exposure policy and external validation; tasks.json valid; v1 suite still green. (commit `d345a92`)
- [x] **D44** — Implement 5-fold patient-grouped stratified CV over train+val; re-run E01 under it as the v2 reference. Deliverable: `src/respiralab/evaluation/group_cv.py, tests, results/E01_cv`. Acceptance: No patient in >1 fold and no test patient in any fold (asserted); per-fold patient/class counts and fold hash recorded; E01 CV mean+-sd reported. (macro_f1=0.3654±0.0369; `reports/v2_group_cv_E01.md`)
- [x] **D45** — Add temperature scaling, ECE and reliability diagram; calibrate E01 on held-out folds. Deliverable: `src/respiralab/evaluation/calibration.py, tests, reports/calibration_E01.md`. Acceptance: Temperature fit on held-out-fold predictions only; ECE before/after; unit test recovers a known temperature on synthetic logits. (T=2.0069; ECE 0.0950→0.0426)
- [x] **D46** — Add 16 kHz common-band (<=1800 Hz) encoder input path with zero|repeat pad option; v1 path untouched. Deliverable: `src/respiralab/dsp/ v2 input variant, tests`. Acceptance: Synthetic tone above 1800 Hz attenuated; fixed output shape; both pad modes tested; v1 preprocess hash 2702dab77672cc55 and results/E01/dev_scores.json reproduce. (`reports/v2_encoder_input.md`)
- [ ] **D47** — Select one pretrained encoder; verify license, weights and input rate; cache frozen embeddings. Deliverable: `src/respiralab/models/encoder.py, embedding cache (gitignored), docs/07_SOURCE_VERIFICATION.md entry`. Acceptance: License, checkpoint hash and input rate recorded (ADR-0001 amended if not 16 kHz); same input gives identical embedding; '~60% ICBHI Score' claim sourced or dropped.
- [ ] **D48** — E05: logistic regression on frozen encoder embeddings under grouped CV. Deliverable: `results/E05, reports/v2_probe.md`. Acceptance: Same folds as results/E01_cv; E06 go/stop decided by the ADR-0001 rule and logged.
- [ ] **D49** — E06: bounded fine-tune of the encoder over 3 seeds under grouped CV (only if D48 = go). Deliverable: `src/respiralab/models/finetune.py, results/E06`. Acceptance: Each run records config/commit/hardware/runtime; selection on held-out folds only; if D48 = stop, closed as a documented skip.
- [ ] **D50** — E07: single-factor 'both'-class ablations - balanced sampling, focal loss, train-only mixup. Deliverable: `results/E07, reports/v2_ablation_both.md`. Acceptance: One factor per run vs class-weighted reference; mixup parents asserted to be train-fold patients; both-recall reported with fold sd.
- [ ] **D51** — E08: repeat-pad vs zero-pad single-factor ablation. Deliverable: `results/E08`. Acceptance: Only padding differs; CV macro-F1 and per-class recall reported.
- [ ] **D52** — Freeze v2 model, temperature and weights hash; write pre-test declaration. Deliverable: `reports/v2_model_freeze.md`. Acceptance: G7 passed; every v2 choice traced to grouped-CV evidence; no primary-test numbers cited.
- [ ] **D53** — Run one-time v2 primary-test evaluation (frozen v2 + calibrated E01), labeled second exposure. Deliverable: `reports/v2_test_results.json`. Acceptance: Run exactly once; patient-bootstrap CIs; labeled post-D29 second exposure per ADR-0001 D4; no subsequent tuning.
- [ ] **D54** — Acquire and audit an external dataset (SPRSound or HF_Lung); write ADR-0002 label mapping. Deliverable: `source-notes.md entry, external inventory manifest, docs/adr/0002-external-label-mapping.md`. Acceptance: User has downloaded the data and recorded its terms; audit counts recorded; label mapping fixed before any prediction.
- [ ] **D55** — Zero-shot external validation of frozen E01 and v2. Deliverable: `reports/external_validation.md`. Acceptance: No tuning of any kind on external data; separate protocol table; population/device shift stated.
- [ ] **D56** — Serve v2: hash-verified versioned weights and a train-fit OOD/input-quality gate in inference and Streamlit. Deliverable: `src/respiralab/inference/, app wiring, tests`. Acceptance: App refuses a mismatched weights hash; OOD threshold fit on train only; rejection path tested; annotated_cycle vs exploratory_window preserved.
- [ ] **D57** — Update model card, final report and release notes v2.0; fresh-clone walkthrough. Deliverable: `reports/model_card.md, reports/final_report.md, release_notes_v2.0.md`. Acceptance: G8 passed; v1 and v2 tables kept separate; no diagnosis claims.

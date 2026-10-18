# RespiraLab — Spec (synced from RespiraLab_Project_Pack)

This file is a build-loop-readable digest of the canonical docs in
`../RespiraLab_Project_Pack/`. Those docs remain the source of truth; this
file is never edited to replan — only re-synced if they change.

**Canonical sources:** `docs/01_PRD.md`, `docs/02_ARCHITECTURE.md`,
`docs/03_DATA_AND_EVALUATION_CONTRACT.md`, `planning/QUALITY_GATES.md`,
`planning/tasks.json`.

## What this is
A reproducible, tested respiratory-cycle sound classifier (Normal / Crackle /
Wheeze / Both) on ICBHI 2017. **Educational research tool, not a diagnostic
device.** Sound-event labels are not disease labels.

## Non-negotiables (do not violate while building)
- Patient-disjoint train/val/test for the primary protocol; official ICBHI
  split is a separate, audited comparison, never assumed disjoint.
- Train-only fit for scalers/augmentation; test stays `test_locked=true`
  until model config/seed/threshold are frozen (Gate G4).
- Primary test evaluated **once**, after freeze — no subsequent tuning on it.
- No fabricated scores; every reported number cites split, n, seed, config.
- Never claim medical diagnosis.

## Status
D01–D42 complete, Gates G1–G6 PASS. v1.0 is published (public repo, live Streamlit
Cloud demo). D43 added `docs/adr/0001-v2-pretrained-encoder-and-evaluation.md`
(required by the data contract: v2 changes sample rate + selection protocol).
D44–D45 done (grouped-CV reference, calibration). v1's frozen numbers and code
paths are untouched by any of this.

## Remaining scope (D46–D57) — v2, per ADR-0001
Phases: Encoder input + selection (D46–D51, ends at Gate G7 model freeze) →
Test + external validation (D52–D55) → Serving (D56) → Launch (D57, Gate G8).

Exact per-task deliverables/acceptance criteria live in `../RespiraLab_Project_Pack/planning/tasks.json`
(D46 onward) — `tasks/plan.md` in this repo mirrors them for the build loop.

## Gates requiring explicit human sign-off (hard stops, not just checklist items)
- **G7 (after D52):** v2 model/temperature/weights frozen on grouped-CV evidence
  only — no primary-test numbers may have informed the freeze.
- **G8 (after D57):** one-time second-exposure test report, external zero-shot
  report, hash-verified serving, updated model card, fresh clone.

## Structural blockers (not code — need a human or external resource)
- **D47** needs a real pretrained-encoder checkpoint selected, its license
  checked, and its actual input sample rate confirmed (ADR-0001 assumes 16 kHz
  pending this). Requires downloading model weights from an external host.
- **D54** needs the user to download SPRSound or HF_Lung and record its license
  terms — "never invent... dataset access" (AGENTS.md). Cannot be done without
  the user's explicit action.

## Stack (already in place, do not re-architect)
Python 3.11 (`uv`), numpy/scipy/soundfile/librosa/scikit-learn, PyTorch,
pytest, Ruff, Streamlit. CLI contract: `python -m respiralab.cli {audit,split,features,train-ml,train-cnn,evaluate,predict}`.
Repo layout under `src/respiralab/{data,dsp,models,evaluation,inference,app}`.

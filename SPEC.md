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

## Remaining scope (D23–D42)
Phases: Deep Learning (D23–D28, ends at Gate G4 model freeze) → Validation
(D29–D34, includes the one-time final test at D29 and Gate G5) → Product
(D35–D40) → Launch (D41–D42, Gate G6).

Exact per-task deliverables/acceptance criteria live in `../RespiraLab_Project_Pack/planning/tasks.json`
(D23 onward) — `tasks/plan.md` in this repo mirrors them for the build loop.

## Gates requiring explicit human sign-off (hard stops, not just checklist items)
- **G4 (after D28):** "DO NOT open final test" until model config/threshold
  frozen. Requires logged go/no-go in `PROGRESS_LOG.md` before D29 runs.
- **G5 (after D34):** one-time test report + model card; no diagnosis claims.
- **G6 (after D42):** fresh-clone reproduction before calling it released.

## Stack (already in place, do not re-architect)
Python 3.11 (`uv`), numpy/scipy/soundfile/librosa/scikit-learn, PyTorch,
pytest, Ruff, Streamlit. CLI contract: `python -m respiralab.cli {audit,split,features,train-ml,train-cnn,evaluate,predict}`.
Repo layout under `src/respiralab/{data,dsp,models,evaluation,inference,app}`.

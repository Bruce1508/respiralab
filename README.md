# RespiraLab

Reproducible respiratory-cycle sound classifier — **Normal / Crackle / Wheeze / Both** — on the ICBHI 2017 dataset.

**This is an educational research tool, not a diagnostic device.** Sound-event labels are not disease labels; no claim is made that any output detects asthma, COPD, or any clinical condition.

## Status
D01 (repository + environment skeleton). See `planning/` in the upstream project pack (`RespiraLab_Project_Pack/`) for the full 42-session execution plan, quality gates, and decision rules.

## Setup
```bash
uv sync
uv run pytest
uv run respiralab --version
```

## Data
Raw ICBHI audio/annotations are **not** committed to this repository. See `source-notes.md` for acquisition, licensing, and provenance. Obtain the dataset yourself from the official host before running any `audit`/`split`/`features`/`train-*` commands.

## Non-negotiables
- No train/val/test patient overlap in the primary evaluation protocol.
- Primary test set is evaluated once, after model selection — never used for tuning.
- No fabricated or inflated metrics; every reported number carries protocol, split, and seed.

Full contracts: `docs/01_PRD.md`, `docs/02_ARCHITECTURE.md`, `docs/03_DATA_AND_EVALUATION_CONTRACT.md` (upstream project pack).

# AI Coding Agent Operating Instructions — RespiraLab

(Canonical copy lives in the upstream `RespiraLab_Project_Pack/AGENTS.md`; this copy travels with the code repo so an agent cloning only this repo still has it.)

## Task selection
Read `README.md` and `source-notes.md` here, and `docs/01_PRD.md`, `docs/03_DATA_AND_EVALUATION_CONTRACT.md`, `planning/tasks.json` in the project pack. Select the first task with status `todo` whose dependencies are `done`; tackle ONLY that task unless a prerequisite defect blocks it.

## Per-task algorithm
1. Restate specific task goal, target artifact, and acceptance check in a short implementation plan.
2. Inspect existing repo and tests, avoid duplicate functionality; choose simplest production-quality design.
3. Implement minimal focused diff; create meaningful unit/integration tests for behavior changed.
4. Execute tests and CLI; capture actual outputs. Never invent file presence, metric values, tests or dataset access.
5. Mark `done` only if acceptance met and evidence exists; otherwise mark `blocked` with exact failure, or `in_progress`.
6. Update progress log: commands, evidence, changed paths, results, hours, next step.
7. Give human reviewer a concise change summary, tests, scientific concerns, and next DXX.

## Critical scientific constraints
- Sound-event labels are not diagnosis labels; never claim asthma detection.
- Patient split must be independent across train/val/test; official benchmark audited separately.
- Never use primary final test for iteration/tuning. Any change in split, label definition, cutoff/sample rate or metric needs an explicit ADR and invalidates prior comparisons if incompatible.
- No fabricated scores or artificially inflated claims. No noise-induced augmentation outside train; no preprocessing fit on holdouts.
- Never commit the real dataset, private audio or secrets. Preserve provenance, hashes, pinned deps, machine specs and config.

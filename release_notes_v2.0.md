# RespiraLab v2.0 — release notes

## Scope

Adds a pretrained-encoder model (AST, bounded fine-tune) and a second,
truly external evaluation (HF_Lung_V1) alongside v1's handcrafted-feature
pipeline. Research/educational tool, **still not a diagnostic device** —
covers v2's roadmap D43-D57 of the project's execution plan, per
`docs/adr/0001-v2-pretrained-encoder-and-evaluation.md`.

## What's included

- Pretrained AST encoder (`MIT/ast-finetuned-audioset-10-10-0.4593`),
  bounded fine-tune of its last 2 blocks, selected over a frozen-embedding
  probe and two other single-factor alternatives on grouped CV alone
  (Gate G7 PASS).
- One-time second exposure of the primary test: v2 macro_f1=0.5208
  vs E01's 0.2861 (D53) — the primary test is now closed for good.
- A genuinely external, zero-shot validation on HF_Lung_V1 (different
  patients, different devices, no tuning of any kind): v2 macro_f1=0.3025
  vs E01's 0.2165 (D54/D55), reported in its own protocol table.
- Hash-verified weight serving from Hugging Face Hub (weights too large
  for git) and a train-fit Mahalanobis OOD gate that refuses to predict on
  off-distribution input, wired into both the inference module and the
  Streamlit demo (Gate G7/G8, D56).
- An updated model card and final report with v1 and v2 results kept in
  separate tables throughout (Gate G8 PASS).

## Known limitations (full detail: `reports/model_card.md`)

v2's OOF-fit calibration temperature *worsens* test ECE instead of
improving it (disclosed, not resolved); external crackle recall is weak
(0.055); external validation for v2 used a stratified 2,000-phase sample,
not the full ~45k-phase corpus; v2 requires network access to Hugging Face
Hub at first load, unlike v1's fully offline, git-committed pipeline. All
of v1's limitations (`both`-class recall, recording clipping, noise
sensitivity, ICBHI Score's blind spot, dataset scope) still apply where
unchanged — they have not been re-studied against v2.

## Verification

- `uv sync && uv run pytest -q` → `247 passed` on a fresh clone (no ICBHI
  data, HF_Lung data, or Hugging Face credentials present) — all
  v2-serving tests (OOD gate fit/accept/reject, hash match/mismatch,
  `annotated_cycle`/`exploratory_window` preservation) are synthetic or
  mocked.
- Real-data/real-network verification (outside the test suite, done once
  each, not repeatable in CI without credentials and large downloads):
  hash-verified download against the live `bruce1508/respiralab-v2` repo
  reproduces the frozen weights' recorded sha256 exactly; the OOD gate
  accepts a real ICBHI cycle (distance=25.57) and correctly rejects a
  synthetic tone (distance=92.54).
- `uv run ruff check src tests` → all checks passed.
- No secrets, no raw audio, and no model weights are committed to git —
  v2's `.pt` weights are gitignored and hash-verified as an external
  artifact, consistent with v1's existing release checklist.

## Not included (explicit non-goals, carried over from v1's PRD)

Clinical diagnosis of any kind, medical device compliance, real-time
hardware integration, state-of-the-art performance claims. Also not yet
done for v2 specifically: fusing E07's balanced_sampling improvement,
and repeating v1's noise-robustness and error-analysis studies against
the new model (flagged as v2.1 candidates, `planning/PROGRESS_LOG.md`).

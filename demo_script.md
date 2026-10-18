# RespiraLab — demo walkthrough

A ~90-second guided walkthrough for a reviewer or recruiter, verified
end-to-end against a genuinely fresh clone (D42).

## 1. Install (fresh clone, no ICBHI data needed)

```bash
git clone https://github.com/Bruce1508/respiralab.git
cd respiralab
uv sync
```

## 2. Prove it's real, not a mockup

```bash
uv run ruff check src tests
uv run pytest -q
```

Expected: `169 passed, 1 skipped` (the 1 skip is the single test that needs
the real ICBHI feature cache, which is gitignored and absent here — see
`tests/inference/test_predict.py`). This proves the reproducible core,
DSP pipeline, models, evaluation, and inference guard rails all run from
nothing but this git history — `tests/integration/` exercises the full
chain on synthetic audio generated at test time (D38).

## 3. Look at the real evidence

- `reports/model_card.md` — what the frozen model is, how it was chosen,
  and every known limitation.
- `reports/final_test_results.json` — the one-time primary test result
  (macro_f1=0.2861), opened exactly once after Gate G4 froze the model.
- `reports/figures/example_*.png` — real waveform/spectrogram figures per
  class, generated from the actual ICBHI pipeline (D10).

## 4. Run the interactive demo (requires your own ICBHI download)

The Streamlit app's prediction needs the real feature cache, since the
frozen model is refit deterministically rather than shipped as a pickle
(see `reports/model_freeze.md`). Follow README.md's "Bring your own
data" section first, then:

```bash
uv run streamlit run src/respiralab/app/streamlit_app.py
```

Upload any mono WAV (≤20MB). You will see:
- the waveform and log-Mel spectrogram, rendered live;
- a persistent "not a diagnostic device" banner;
- the 4-class probability bar chart, with `analysis_mode` shown as
  `exploratory_window` (no cycle bounds given) or `annotated_cycle` (if
  you provide expert-style bounds) — the UI never calls its output a
  diagnosis.

## 5. The one sentence that matters

Every number in this repo cites a commit, a seed, and a split hash — the
full accountability trail is `planning/PROGRESS_LOG.md` in
`RespiraLab_Project_Pack` and the `git log` on this repo.

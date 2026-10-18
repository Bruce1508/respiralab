# Fresh-clone smoke test

`tests/integration/test_pipeline_smoke.py` exercises the full pipeline
(ingest → patient-level split → feature extraction → train → predict,
plus the offline-inference guard rails from D35) using **only synthetic
sine-tone WAVs generated at test time** — no ICBHI raw audio is required
or shipped in this repository.

Run it on its own:

```bash
uv run pytest tests/integration/ -v
```

Or as part of the full suite (`uv run pytest -q`) — it runs alongside
every other test, no special setup, no network access, no real dataset.

This proves a fresh clone (without `data/raw/ICBHI_final_database/`,
which is gitignored and never committed) can still install, import, and
exercise every pipeline stage end-to-end.

# D39 — Release checklist (CI + sensitive-data scan)

## CI (`.github/workflows/ci.yml`)
Runs on every push/PR: `uv sync` → `ruff check` → package import smoke →
CLI smoke → full test suite (`uv run pytest -q`, synthetic-only per D38,
no real ICBHI data needed or present in CI).

All 4 steps verified locally before commit (same commands CI runs):

| Check | Result |
|---|---|
| `uv run ruff check src tests` | All checks passed |
| `uv run python -c "import respiralab; print(respiralab.__version__)"` | `0.1.0` |
| `uv run respiralab --version` | `respiralab 0.1.0` |
| `uv run pytest -q` | 170 passed |

## Sensitive-data scan (git-tracked files only)

- **Files >1MB:** none (`git ls-files` + `du` scan, sorted descending — empty result).
- **Secret-like patterns** (`api_key`, `secret`, `password`, `aws_access`, `private_key`, PEM headers): one match, in `AGENTS.md`'s own policy text ("Never commit... secrets") — not an actual credential.
- **Raw audio** (`.wav`/`.mp3`): zero tracked files. `data/raw/`, `data/interim/`, `data/processed/`, and `artifacts/` (including the real feature cache) are all gitignored, confirmed via `.gitignore`.
- **Model artifacts:** none committed — this project's established pattern (D15-D28) is to refit the frozen pipeline deterministically from code+config+split rather than persist a pickle (see `reports/model_freeze.md`).

## Release blocker criteria (`docs/05_TESTING_AND_SECURITY.md`) — checked

- Patient leakage in primary split? No (D07, re-verified D28).
- Test used in tuning? No (D29 is one-time, closed).
- Inappropriate clinical/diagnosis claims? No (model card D34, explicit throughout).
- Unsafe file upload processing? No (D35/D36: in-memory only, 20MB cap, typed rejection).
- Missing reproducibility info? No (commit/seed/hash recorded on every result since D15).
- Metric calculation untested? No (`evaluation/` has hand-calculated + sklearn-cross-checked tests since D18).
- Deployment storing sensitive audio without consent? No — Streamlit app never persists uploads.

**No release blockers found.**

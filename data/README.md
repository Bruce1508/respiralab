# Dataset acquisition — ICBHI 2017 Respiratory Sound Database

**Do this manually; nothing in this repo downloads the dataset for you, and the real
archive must never be committed (`data/raw/` is gitignored).**

## 1. Get the archive
1. Go to the official challenge page: https://bhichallenge.med.auth.gr/node/50
2. Read the page's stated terms of use / license **before downloading** — do not assume
   redistribution rights beyond what the page states. Quote the exact terms into
   `source-notes.md` under "Acquisition status" once you've read them.
3. Download the official archive (the page describes it as a single package covering
   all 920 recordings + annotation `.txt` files + supporting metadata). Note the exact
   filename and any version/date shown on the page — don't assume a filename not shown
   on the page itself.
4. Record the retrieval date and the archive's own sha256 (e.g. `shasum -a 256 <archive>`)
   in `source-notes.md`.

## 2. Extract
Extract the archive into `data/raw/` at the repo root, e.g.:
```bash
mkdir -p data/raw
unzip /path/to/downloaded_archive.zip -d data/raw
```
`data/raw/` is already in `.gitignore` — extracting here is safe, it will not be committed.

## 3. Verify what you actually got (don't trust the official counts blindly)
Run the inventory CLI added in D02. It hashes and sizes every file under `data/raw/`
and writes a CSV manifest, without reading or printing audio content:
```bash
uv run python -m respiralab.data.inventory --dir data/raw --out manifests/dataset_inventory_raw.csv
```
- Exit code `0` = every file was readable; nonzero = failures were printed to stderr.
- Cross-check the row count against the official page's claim (920 `.wav` files + 920
  paired `.txt` annotation files = 1840, plus whatever extra metadata files the archive
  includes — e.g. a patient-diagnosis file and an official train/test split file).
  **Discrepancies are expected findings to document, not something to silently ignore
  or force to match** (`docs/03_DATA_AND_EVALUATION_CONTRACT.md`).
- WAV header validation (duration, sample rate, channel count, corrupt-file detection)
  is D03, not this step — this CLI only checks that files exist, their size, and their
  checksum.

## 4. What not to do
- Do not commit anything under `data/raw/`, `data/interim/`, or `data/processed/`.
- Do not redistribute the archive or individual recordings without having read and
  satisfied the official terms of use.
- Do not treat the manifest produced here as proof of data integrity — it only proves
  the files you have are byte-identical to themselves on re-hash, not that they're the
  correct/complete official dataset.

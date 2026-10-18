# Dataset source and licensing decision — D01

## Chosen source
**ICBHI 2017 Respiratory Sound Database**, official host: https://bhichallenge.med.auth.gr/node/50

Expected contents (per official page, to be independently verified against the actual downloaded archive at acquisition time, not assumed): 920 recordings, 126 subjects, 6,898 respiratory cycles, ~5.5 hours total audio, sample rates 4 kHz/10 kHz/44.1 kHz depending on recording device.

## Acquisition status
**Downloaded 2026-10-18.** Retrieved via browser from the official page above; the browser auto-extracted the zip on download and did not preserve the original archive file, so an archive-level sha256 could not be captured. Provenance instead rests on: (a) the official source URL, (b) today's retrieval date, (c) a full per-file sha256 manifest of the extracted contents (`manifests/dataset_inventory_raw.csv`, produced by `respiralab.data.inventory`, not committed pending the license review below).

Exact license/terms-of-use text has **not yet been located as a separate document** on the official page — the page itself only states the database "is made freely available for research." No explicit redistribution grant (e.g. CC license) was found. Licensing stance below remains the operative assumption until a more specific terms page is found or the authors are contacted.

Extracted to `data/raw/ICBHI_final_database/` (gitignored, not committed).

### Audit findings (2026-10-18, via `respiralab.data.inventory` + ad hoc soundfile/annotation check — not yet the formal D03/D04 modules)
- File counts: 920 `.wav` + 920 paired `.txt` annotation files + `filename_differences.txt` + `filename_format.txt` + `script.sh` (rename helper) = 1843 files total. Every `.wav` has exactly one matching `.txt` and vice versa — zero orphans.
- Patient IDs: 101–226, **126 unique** — matches official claim exactly.
- WAV headers (via `soundfile.info`, metadata only): all 920 files readable, all mono (1 channel). Sample rates: 44100 Hz (824 files), 4000 Hz (90 files), 10000 Hz (6 files) — confirms the documented 4k/10k/44.1k mix (`docs/03_DATA_AND_EVALUATION_CONTRACT.md` Nyquist warning applies). Duration range 7.86s–86.20s (official page states "10s to 90s" as approximate — minor discrepancy noted, not forced to match).
- Annotation parsing (ad hoc, all 920 `.txt` files, 4-column `start end crackle wheeze` format): 6898 total cycles, **exact match** to official per-class counts — normal=3642, crackle=1864, wheeze=886, both=506. Zero malformed lines.
- Not present in this archive: the separate diagnosis file and official train/test split file that the page links to under "can be found here" (two more downloads) — needed for D06 (official-split overlap audit) and general EDA, not yet acquired.
- This was a verification pass, not the formal D03 (WAV header CSV + corrupt-file report) or D04 (annotation parser + unit tests) deliverables — those still need to be built as real modules with tests, reusing these findings as the expected baseline to test against.

## Licensing stance
- Dataset is obtained directly from the official challenge host, not a third-party mirror.
- Terms of use will be read and quoted (not inferred) before any redistribution decision; default assumption is **no redistribution rights** for raw audio unless the official terms explicitly grant them.
- Raw audio and per-patient metadata are never committed to this git repository (`data/raw/` is gitignored). Only derived, aggregate, non-identifying artifacts (manifests, hashes, feature caches, split IDs where the license permits) may be committed.

## Known caveats to carry forward
- Two patient IDs (156, 218) have been reported by secondary sources as overlapping the *official* ICBHI train/test split. This must be independently audited against the actual downloaded split file (D06) before relying on the official split as patient-disjoint — the primary protocol in this project uses its own frozen patient-disjoint split (D07) regardless.
- Diagnosis labels present in the official metadata (e.g. COPD, asthma) are out of scope for this project's modeling target and must never be used as training labels; only the per-cycle crackle/wheeze annotations are the modeling target (`docs/01_PRD.md`, `docs/03_DATA_AND_EVALUATION_CONTRACT.md`).

## Next action
D02: write `data/README.md` download instructions and the checksum/inventory CLI, then actually acquire the archive and fill in retrieval date + sha256 above.

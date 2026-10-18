# Dataset source and licensing decision — D01

## Chosen source
**ICBHI 2017 Respiratory Sound Database**, official host: https://bhichallenge.med.auth.gr/node/50

Expected contents (per official page, to be independently verified against the actual downloaded archive at acquisition time, not assumed): 920 recordings, 126 subjects, 6,898 respiratory cycles, ~5.5 hours total audio, sample rates 4 kHz/10 kHz/44.1 kHz depending on recording device.

## Acquisition status
**Not yet downloaded.** This file records the *decision*, not a completed download. Retrieval date, archive sha256, and exact license/terms-of-use text must be appended here at acquisition time (D02), per `docs/03_DATA_AND_EVALUATION_CONTRACT.md`.

## Licensing stance
- Dataset is obtained directly from the official challenge host, not a third-party mirror.
- Terms of use will be read and quoted (not inferred) before any redistribution decision; default assumption is **no redistribution rights** for raw audio unless the official terms explicitly grant them.
- Raw audio and per-patient metadata are never committed to this git repository (`data/raw/` is gitignored). Only derived, aggregate, non-identifying artifacts (manifests, hashes, feature caches, split IDs where the license permits) may be committed.

## Known caveats to carry forward
- Two patient IDs (156, 218) have been reported by secondary sources as overlapping the *official* ICBHI train/test split. This must be independently audited against the actual downloaded split file (D06) before relying on the official split as patient-disjoint — the primary protocol in this project uses its own frozen patient-disjoint split (D07) regardless.
- Diagnosis labels present in the official metadata (e.g. COPD, asthma) are out of scope for this project's modeling target and must never be used as training labels; only the per-cycle crackle/wheeze annotations are the modeling target (`docs/01_PRD.md`, `docs/03_DATA_AND_EVALUATION_CONTRACT.md`).

## Next action
D02: write `data/README.md` download instructions and the checksum/inventory CLI, then actually acquire the archive and fill in retrieval date + sha256 above.

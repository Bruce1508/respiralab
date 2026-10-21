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

---

# External dataset source and licensing decision — D54

## Chosen source
**HF_Lung_V1**, Heroic-Faith Medical Science Co. Ltd. Official host: https://gitlab.com/techsupportHF/HF_Lung_V1

Per ADR-0001 D6: the external dataset is v2's headline generalization evidence (the only holdout
nobody — not even this project's own design choices — has looked at). SPRSound (pediatric) was
the other candidate considered; HF_Lung was chosen for its unambiguous CC BY 4.0 license (no
competition/registration portal) and larger event-level label set. SPRSound may still be acquired
later under its own ADR (ADR-0001 D6's "own protocol" rule) if time allows — not a blocker for D54.

## Acquisition status
**Downloaded by the user 2026-10-21**, extracted via `p7zip` (`train.7z.001-010`, `test.7z.001-003`)
into `data/external/HF_Lung_V1_{train,test}/` (gitignored, not committed, per this file's own
licensing stance below and AGENTS.md). Repo snapshot dated 2023-03-06 per the archive's own
`README.md`/`LICENSE`/`Disclaimer`, copied alongside the data at `data/external/{README.md,LICENSE,Disclaimer}`
(also gitignored — reproducible from the public repo at any time).

### Audit findings (2026-10-21, via `respiralab.data.inventory` + an ad hoc label-line parser — not yet a formal module, same status as D01's initial pass)
- File counts: 7,809 `.wav` + 7,809 paired `.txt` (train) and 1,956 `.wav` + 1,956 paired `.txt`
  (test) = **9,765 `.wav` total, exact match to the archive's own README claim.** Zero orphans
  (every wav has exactly one label file and vice versa), zero unreadable files
  (`manifests/hf_lung_{train,test}_inventory.csv`).
- Label-line parsing (ad hoc, all 9,765 `.txt` files, `<type> <start> <end>` format): zero
  malformed lines. Event-type counts (I=34,095, E=18,349, D/crackle=15,606, Wheeze=8,457,
  Rhonchi=4,740, Stridor=686) **match the README's published totals exactly in every category** —
  strong evidence the download and audit are both correct (`manifests/hf_lung_label_audit.json`).
- This dataset labels **breath phases** (I/E), not whole respiratory cycles like ICBHI — see
  `docs/adr/0002-external-label-mapping.md` for the full label-mapping decision this forces.
- Subject count: the README claims 261 (TSECC) + 18 (RCW/RCC) = 279 source patients. A same-date
  filename-token proxy gives 130 (train) + 41 (test) unique dates — **not** treated as a verified
  per-patient count, only as a rough cross-check, per the README's own "very likely same subject"
  hedge.
- Population: **adult** (TSECC = emergency/critical-care database; RCW/RCC = respiratory-care-ward
  residents) — not pediatric, unlike SPRSound. Device: digital stethoscope (Littmann 3200,
  `steth_` prefix) and a custom multichannel acoustic device (`HF_Type-1`, `trunc_` prefix) —
  neither matches any of ICBHI's three native devices.

## Licensing stance
- **CC BY 4.0** (Creative Commons Attribution 4.0 International), full text in
  `data/external/LICENSE`. Requires attribution; permits redistribution of adapted material
  under compatible terms — materially more permissive than ICBHI's ambiguous "freely available
  for research" stance (D01 above).
- This project still does **not** redistribute the raw audio regardless of what the license
  permits, for consistency with its ICBHI stance: `data/external/` is gitignored; only derived,
  aggregate, non-identifying artifacts (file-inventory manifests, label-audit counts, eventual
  zero-shot metrics) are ever committed.
- The dataset's own `Disclaimer` states its labels are "not ground truth" and the provider
  disclaims responsibility for label accuracy or any resulting loss — carried forward as an
  explicit caveat in every report that uses this data (D55's `reports/external_validation.md`),
  never silently assumed away.

## Known caveats to carry forward
- Breath-phase vs. whole-cycle unit mismatch against ICBHI (see ADR-0002) — external-validation
  numbers are **not** directly comparable to ICBHI macro-F1 figures and must never be placed in
  the same table (ADR-0001 D6).
- Rhonchi/Stridor events have no established equivalence to ICBHI's wheeze category and are
  **excluded**, not folded in, from the primary 4-class mapping (ADR-0002) — reported as a
  separate excluded-phase count.
- Population and device are both different from ICBHI; this is deliberate (it's what makes this
  a real zero-shot generalization test) but must be stated plainly in any report, never implied
  to be an apples-to-apples comparison.

## Next action
D55: zero-shot external validation of frozen E01 and v2 against this label mapping — implement
the parser per ADR-0002, write it with unit tests (not just ad hoc checks), run it once with no
tuning, write `reports/external_validation.md` as its own protocol table.

# ADR-0002 — HF_Lung_V1 external label mapping (D54, per ADR-0001 D6)

- **Status:** Accepted.
- **Date:** 2026-10-21
- **Required by:** ADR-0001 D6 — "the mapping... [is] fixed in ADR-0002 before any prediction is run." No prediction code is run against this dataset until this ADR is accepted.
- **Scope:** HF_Lung_V1 only. If SPRSound is acquired later, it gets its own ADR (ADR-0003), not an amendment here, per ADR-0001 D6's "its own protocol" rule.

## Context

HF_Lung_V1 (Heroic-Faith Medical Science Co. Ltd., CC BY 4.0) is an **event-level** lung-sound
database: 9,765 15-second `.wav` recordings, each with a plain-text label file whose lines are
`<TYPE> <start HH:MM:SS.mmm> <end HH:MM:SS.mmm>`. `TYPE` is one of:

| Type | Meaning | Train count | Test count | Total | Matches README? |
|---|---|---:|---:|---:|---|
| `I` | Inhalation (breath phase) | 27,223 | 6,872 | 34,095 | Yes |
| `E` | Exhalation (breath phase) | 15,601 | 2,748 | 18,349 | Yes |
| `D` | Discontinuous adventitious sound (= crackle) | 13,794 | 1,812 | 15,606 | Yes |
| `Wheeze` | Continuous adventitious sound | 7,027 | 1,430 | 8,457 | Yes |
| `Rhonchi` | Continuous adventitious sound | 3,780 | 960 | 4,740 | Yes |
| `Stridor` | Continuous adventitious sound | 657 | 29 | 686 | Yes |

(Counts reproduced in this session from the actual downloaded archive, not copied from the
README — they match it exactly. See "Audit evidence" below.)

This is fundamentally different from ICBHI's labeling: ICBHI annotates whole **respiratory
cycles** (inhale+exhale as one unit) with crackle/wheeze presence. HF_Lung_V1 has no explicit
cycle boundary — only alternating `I`/`E` breath-phase intervals, with adventitious-sound events
that temporally overlap them (confirmed: `D` events usually, but not always, span exactly the
timestamps of their enclosing `I`/`E` phase — 1,570/1,812 = 86.6% exact match in the test split;
the rest overlap partially). **No cycle boundary may be invented that the data doesn't contain.**

## Decision

### Unit of analysis: the breath phase, not a reconstructed "cycle"

Each `I` or `E` interval is treated as one cycle-equivalent unit — the finest unit this dataset
actually demarcates. Full inhale+exhale "cycles" are **not** reconstructed by pairing adjacent
I/E events, because gaps between them are common and pairing them would invent structure the
annotators didn't provide. This is reported explicitly as a **unit-of-analysis mismatch** against
ICBHI's whole-cycle unit — not glossed over.

### Label assignment per breath-phase unit

An adventitious event "belongs" to a breath-phase interval if their time spans **overlap at all**
(not exact-match, since D events aren't always exactly coincident with their phase).

| Overlapping adventitious types | Assigned 4-class label |
|---|---|
| None | `normal` |
| `D` only | `crackle` |
| `Wheeze` only | `wheeze` |
| `D` and `Wheeze` | `both` |
| `Rhonchi` and/or `Stridor`, with no `D`/`Wheeze` overlap | **excluded** (see below) |

### Rhonchi and Stridor are excluded from the primary mapping, not folded into `wheeze`

Rhonchi and Stridor are real, distinct adventitious sound categories (low-pitched
secretion-related sound; upper-airway sound, respectively) — not interchangeable with wheeze by
definition, and ICBHI's own protocol has no equivalent category to map them to. Silently lumping
them into `wheeze` would fabricate an equivalence the clinical literature doesn't support, which
AGENTS.md's "no fabricated... inflated claims" rule forbids. Phases whose *only* overlapping
adventitious events are Rhonchi/Stridor are **excluded from the primary 4-class accuracy numbers**
and reported as a separate count (how many phases, what fraction) in `reports/external_validation.md`
(D55) — not silently dropped, not silently included.

### No cycle padding/filtering rules are imported from ICBHI

v1/v2's duration-based cycle padding (`pad_or_crop`, D08/E08's zero-pad) applies to ICBHI cycles.
HF_Lung phases get their own padding to the AST encoder's input window, following the already-
frozen v2 encoder-input pipeline (`dsp/encoder_input.py`) unchanged — this ADR fixes the *label*
mapping only, not a new preprocessing path.

### No tuning, no threshold/temperature changes

Per ADR-0001 D6: this dataset gets zero-shot inference from the already-frozen E01 (Gate G4) and
v2 (Gate G7) models only. No fine-tuning, no threshold search, no temperature refitting. Results
are a **separate protocol table**, never merged into the ICBHI tables, with population
(pediatric vs. adult — actually HF_Lung's TSECC source is adult/ED patients, RCW/RCC source is
adult long-term-care residents; HF_Lung is **not** pediatric, unlike SPRSound) and device shift
(digital stethoscope Littmann 3200 + a custom multichannel acoustic device `HF_Type-1`, vs.
ICBHI's own device mix) stated explicitly in D55's report.

## Audit evidence (2026-10-21, ad hoc check via Python, not yet a formal module)

- Acquired from https://gitlab.com/techsupportHF/HF_Lung_V1 (train+test `.7z` archives,
  2023-03-06 repo snapshot), extracted to `data/external/HF_Lung_V1_{train,test}/` (gitignored,
  not committed, per AGENTS.md).
- File-level inventory (sha256 + size + readability) via the existing
  `respiralab.data.inventory` CLI: `manifests/hf_lung_{train,test}_inventory.csv`. 15,618 + 3,912
  files total (7,809 `.wav` + 7,809 `.txt` train; 1,956 `.wav` + 1,956 `.txt` test = **9,765 `.wav`
  files total, exact match to the README's claim**). Zero unreadable files.
- Every `.wav` has exactly one matching `_label.txt` and vice versa in both splits — zero orphans.
- Zero malformed label lines (every line parses as `<type> <start> <end>`).
- Event-type counts (table above) match the README's published totals **exactly** in every
  category — strong evidence the download and audit are both correct.
- Device split: `steth_` (Littmann 3200) 3,254 train + 1,250 test; `trunc_` (HF_Type-1) 4,555
  train + 706 test.
- Unique same-date tokens (a loose subject-count proxy per the README's own caveat that
  same-date files are "very likely" the same subject — **not** a verified patient ID): 130
  (train), 41 (test). This is reported as a proxy, not asserted as an exact patient count; the
  README's own text is the only subject-count claim this project treats as authoritative
  (261 TSECC + 18 RCW/RCC = 279 source patients, not independently re-derivable from filenames
  alone).

## License

CC BY 4.0 (Heroic-Faith Medical Science Co. Ltd.), full text in `data/external/LICENSE`
(gitignored alongside the data it accompanies, since it's bundled with the archive — the
license *text itself* is reproducible from the public repo at any time, so losing the local
copy loses no information). Requires attribution; permits redistribution of adapted material
under compatible terms. No redistribution of the raw audio is planned by this project regardless
(same stance as ICBHI, per `source-notes.md`) — only derived, non-identifying artifacts
(manifests, aggregate metrics) may ever be committed.

The dataset's own `Disclaimer` states its labels are "not ground truth" and the data provider
disclaims responsibility for label accuracy — carried forward into D55's report as a stated
limitation of this external-validation protocol, not silently assumed away.

## Consequences

- D55 (zero-shot external validation) implements a parser following exactly the rule above:
  group a label file's lines into `I`/`E` phase intervals, overlap-match `D`/`Wheeze` events onto
  them, exclude Rhonchi/Stridor-only phases, emit one 4-class label per phase alongside its audio
  span — with unit tests against this ADR's rule (not just against the ad hoc audit numbers
  above).
- `reports/external_validation.md` must report, per ADR-0001 D6: the four class counts,
  the excluded-phase count/fraction, population/device shift, and "not ground truth" from the
  dataset's own disclaimer — none of this may be smoothed into a single comparable macro-F1
  table next to ICBHI's.

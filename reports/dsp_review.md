# DSP phase review — Gate G2 (D14)

## Scope
Everything built D08–D13: deterministic cycle extraction, sample-rate-safe
resampling, optional filtering, diagnostic plotting, handcrafted features,
feature caching with lineage hashes.

## Full regression + build
- `uv run pytest -q` → **108 passed**, 0 failed, 0 skipped.
- `uv run ruff check src tests` → **All checks passed**.
- All 5 committed DSP report figures (`example_normal/crackle/wheeze/both.png`,
  `filter_before_after.png`) verified non-corrupt (`PIL.Image.verify()`), all
  render at their expected size.

## Abnormal / edge-case clips (real data, not synthetic)
Ran the full crop → resample(4000Hz) → bandpass(100–1800Hz) → extract_features
pipeline on the most extreme real cycles found:

| case | cycle_id | native_sr | cropped samples | resampled samples | filter | feature shape | finite |
|---|---|---|---|---|---|---|---|
| shortest cycle (0.200s, dataset min) | `217_1b1_Tc_sc_Meditron_c010` | 44100 | 8820 | 800 | OK | (46,) | yes |
| longest cycle (16.163s, dataset max) | `166_1p1_Al_sc_Meditron_c005` | 4000 | 64652 | 64652 | OK | (46,) | yes |
| most-clipped recording's first cycle | `130_2p3_Pl_mc_AKGC417L_c000` | 44100 | 48290 | 4381 | OK | (46,) | yes |

No exceptions, no non-finite values, no shape deviation on any of these —
including the 0.2s cycle, which at 4000Hz resampled rate is only 800 samples
(well under naive assumptions about minimum usable audio length).

## New finding this session: dataset-wide clipping
A full scan of all 920 recordings (not sampled) found:
- **0** near-silent recordings (min RMS across the dataset: 0.0024 — the
  silence-handling unit tests on synthetic all-zero audio were precautionary,
  not reflecting a real-data case).
- **177 of 920 recordings (19.2%)** have >1% of samples sitting at full-scale
  amplitude (±1.0) — i.e. likely clipped at the source, not a pipeline
  artifact. Worst case: `130_2p3_Pl_mc_AKGC417L.wav` at 48.2% of samples
  clipped.

This is a genuine dataset characteristic, not a defect in this project's
code — the pipeline handles clipped audio without crashing or producing
non-finite features (confirmed above), but clipping is real signal
distortion that should be carried forward as a documented **limitation**
(model card, D34) rather than silently ignored. It may also explain some of
the device-specific confounding `docs/03_DATA_AND_EVALUATION_CONTRACT.md`
already flags as a risk to inspect.

## Full-dataset evidence (already gathered in D12/D13, referenced not re-run)
- D12: all 6898 real cycles → single feature shape (46), 0 non-finite values.
- D13: full feature cache built (6898/6898), determinism re-verified via a
  20-cycle independent re-run with 0 hash mismatches.
- D09: resampling verified correct on all 3 native sample rates (44100/10000/4000Hz).

## Go/no-go questions (QUALITY_GATES.md)
1. **Are all numbers derived from actual execution artifacts, not stated
   ambitions?** Yes — every count/timing/hash in this report and in
   D08–D13's progress log entries came from an actual command run against
   real data, not an estimate.
2. **Is the underlying model being trained for sound labels, not medical
   disease?** N/A yet (no model trained this phase) — the label map
   (`normal/crackle/wheeze/both`) built in D04 contains no diagnosis labels,
   and `docs/03`'s constraint carries forward unchanged.
3. **Are any subjects, feature fit operations, or augmentations leaking
   between sets?** No feature *fitting* happens in this phase (D12's
   features are fit-free by construction); D13 explicitly tests that
   feature computation is split-independent (same audio → same hash
   regardless of train/test tag). No augmentation exists yet (D25).
4. **Can a second developer reproduce the evidence without copying your
   notebook state?** Yes — every number here came from a `uv run python -m
   respiralab.dsp.*` CLI or a short documented script; no notebook state is
   involved anywhere in this project.
5. **Would showing this to an ML engineer produce defensible answers about
   limitations?** Yes, with the clipping finding now explicitly documented
   rather than hidden — the honest answer to "did you check for clipping"
   is now "yes, 19.2% of recordings, here's the exact scan."

## Gate G2 decision: **PASS**
Deterministic signal pipeline confirmed (D08/D09 hash/output-length checks),
Nyquist-safe resampling and filtering confirmed (D09/D11 anti-alias tests +
real before/after inspection), finite fixed-shape features confirmed at full
dataset scale (D12/D13), 5 clean example figures committed, full unit test
suite green. ML work (D15+) may proceed. The clipping finding is carried
forward as a documented limitation, not a blocker.

# D46 — 16 kHz common-band encoder input path

**Module:** `src/respiralab/dsp/encoder_input.py` (v2-only; reuses v1's existing
`dsp/resample.py::resample_audio` and `dsp/filters.py::lowpass_filter` rather than duplicating
them). Per ADR-0001 D2: resample to 16 kHz (the common pretrained-encoder input rate), low-pass at
1800 Hz (v1's upper band edge, below every native source's Nyquist) so content only present on
10k/44.1k devices cannot become a device shortcut, then pad or crop to a fixed 6.0 s window with a
selectable `zero` or `repeat` pad mode (`pad_or_crop`, needed for the D51 padding ablation).

## Acceptance evidence

- **Synthetic tone above 1800 Hz attenuated:** a 3000 Hz tone (real content only on 44.1 kHz
  sources) loses >90% of its energy after `prepare_encoder_input` vs. the same tone merely
  resampled (`tests/dsp/test_encoder_input.py::test_tone_above_common_band_cutoff_is_attenuated`).
  A 400 Hz tone keeps >50% of its energy (`..._is_preserved`).
- **Fixed output shape:** confirmed for 4 kHz/10 kHz/44.1 kHz inputs of different native durations —
  all return exactly `16000 * 6.0 = 96000` samples.
- **Both pad modes tested:** `zero` pads with zeros past the real signal; `repeat` tiles the signal
  (both also tested for the crop case, where they're identical).
- **v1 untouched:** `uv run python -m respiralab.models.traditional ...` reproduces
  `results/E01/dev_scores.json` bit-for-bit (preprocess hash `2702dab77672cc55`, unchanged) after
  adding this module — no v1 code path was modified.

## Verification

- `uv run pytest -q` -> 190 passed (182 prior + 8 new in `tests/dsp/test_encoder_input.py`).
- `uv run ruff check src tests` -> All checks passed.

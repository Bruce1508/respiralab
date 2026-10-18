"""D09 tests (RED first): sample-rate-safe resampling — correct output
length, anti-aliasing on downsample, no-op when rates already match.
"""

import numpy as np
import pytest

from respiralab.dsp.resample import ResampleError, resample_audio


def test_output_length_matches_resample_poly_formula_downsample():
    orig_sr, target_sr = 44100, 4000
    audio = np.zeros(orig_sr * 2, dtype=np.float32)  # 2s
    out = resample_audio(audio, orig_sr, target_sr)
    # scipy.signal.resample_poly's own length formula: ceil(len(x) * up / down)
    import math

    g = math.gcd(target_sr, orig_sr)
    up, down = target_sr // g, orig_sr // g
    expected_len = math.ceil(len(audio) * up / down)
    assert len(out) == expected_len


def test_output_length_matches_formula_upsample():
    orig_sr, target_sr = 4000, 10000
    audio = np.zeros(orig_sr * 1, dtype=np.float32)
    out = resample_audio(audio, orig_sr, target_sr)
    import math

    g = math.gcd(target_sr, orig_sr)
    up, down = target_sr // g, orig_sr // g
    expected_len = math.ceil(len(audio) * up / down)
    assert len(out) == expected_len


def test_same_rate_is_a_noop():
    audio = np.random.default_rng(0).standard_normal(4000).astype(np.float32)
    out = resample_audio(audio, 4000, 4000)
    np.testing.assert_array_equal(out, audio)


def test_rejects_nonpositive_target_rate():
    audio = np.zeros(1000, dtype=np.float32)
    with pytest.raises(ResampleError):
        resample_audio(audio, 4000, 0)


def test_rejects_nonpositive_orig_rate():
    audio = np.zeros(1000, dtype=np.float32)
    with pytest.raises(ResampleError):
        resample_audio(audio, -4000, 4000)


def test_downsample_attenuates_energy_above_new_nyquist():
    """Anti-alias check: a tone above the target Nyquist must not reappear
    as strong low-frequency energy (aliasing) in the downsampled signal."""
    orig_sr, target_sr = 44100, 4000
    duration_s = 1.0
    t = np.arange(int(orig_sr * duration_s)) / orig_sr
    # a tone well above target_sr/2 (2000 Hz) that would alias into a low
    # frequency if resampled naively (e.g. by naive decimation without a filter)
    tone_freq = 18000.0
    audio = np.sin(2 * np.pi * tone_freq * t).astype(np.float32)

    out = resample_audio(audio, orig_sr, target_sr)

    # compute the output spectrum; with correct anti-aliasing, total energy
    # should have been strongly suppressed (the tone is filtered out, not
    # aliased down into the passband)
    spectrum = np.abs(np.fft.rfft(out))
    input_spectrum = np.abs(np.fft.rfft(audio))
    # energy should drop by a large factor since the tone is entirely removed,
    # not aliased into a lower frequency that preserves its power
    assert spectrum.max() < input_spectrum.max() * 0.1

from __future__ import annotations

import numpy as np
import pytest

from respiralab.dsp.encoder_input import (
    ENCODER_SAMPLE_RATE,
    TARGET_DURATION_S,
    pad_or_crop,
    prepare_encoder_input,
)


def _tone(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float64)


def _energy(signal: np.ndarray) -> float:
    return float(np.sum(signal**2))


def test_output_has_fixed_shape_regardless_of_input_rate():
    for orig_sr, duration in [(4000, 1.0), (10000, 3.0), (44100, 0.5)]:
        audio = _tone(100.0, duration, orig_sr)
        out = prepare_encoder_input(audio, orig_sr)
        assert out.shape == (int(ENCODER_SAMPLE_RATE * TARGET_DURATION_S),)


def test_tone_above_common_band_cutoff_is_attenuated():
    # 44.1kHz source has real content at 3000Hz; after resample+lowpass(1800Hz) it should drop.
    orig_sr = 44100
    high_tone = _tone(3000.0, 2.0, orig_sr)
    out = prepare_encoder_input(high_tone, orig_sr)
    # compare against the same tone just resampled (no filter) to isolate the filter's effect
    from respiralab.dsp.resample import resample_audio

    resampled_only = resample_audio(high_tone, orig_sr, ENCODER_SAMPLE_RATE)
    assert _energy(out[: len(resampled_only)]) < _energy(resampled_only) * 0.1


def test_tone_below_common_band_cutoff_is_preserved():
    orig_sr = 44100
    low_tone = _tone(400.0, 2.0, orig_sr)
    out = prepare_encoder_input(low_tone, orig_sr)
    from respiralab.dsp.resample import resample_audio

    resampled_only = resample_audio(low_tone, orig_sr, ENCODER_SAMPLE_RATE)
    assert _energy(out[: len(resampled_only)]) > _energy(resampled_only) * 0.5


def test_zero_pad_mode_pads_with_zeros():
    short_audio = np.ones(100, dtype=np.float64)
    out = pad_or_crop(short_audio, target_n_samples=1000, mode="zero")
    assert len(out) == 1000
    assert (out[100:] == 0).all()
    assert (out[:100] == 1).all()


def test_repeat_pad_mode_tiles_the_signal():
    short_audio = np.array([1.0, 2.0, 3.0])
    out = pad_or_crop(short_audio, target_n_samples=10, mode="repeat")
    assert len(out) == 10
    np.testing.assert_array_equal(out, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1])


def test_pad_or_crop_crops_when_longer_than_target():
    audio = np.arange(20, dtype=np.float64)
    out_zero = pad_or_crop(audio, target_n_samples=5, mode="zero")
    out_repeat = pad_or_crop(audio, target_n_samples=5, mode="repeat")
    np.testing.assert_array_equal(out_zero, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(out_repeat, [0, 1, 2, 3, 4])


def test_repeat_pad_raises_on_empty_audio():
    with pytest.raises(ValueError):
        pad_or_crop(np.array([]), target_n_samples=10, mode="repeat")


def test_unknown_pad_mode_raises():
    with pytest.raises(ValueError):
        pad_or_crop(np.ones(5), target_n_samples=10, mode="bogus")  # type: ignore[arg-type]

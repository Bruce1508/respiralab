"""D11 tests (RED first): optional high/low/bandpass filters — correct
passband behavior, sample-rate-aware cutoff validation, invalid cutoffs
rejected rather than silently clamped.
"""

import numpy as np
import pytest

from respiralab.dsp.filters import FilterError, bandpass_filter, highpass_filter, lowpass_filter


def _tone(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _energy(signal: np.ndarray) -> float:
    return float(np.sum(signal.astype(np.float64) ** 2))


def test_lowpass_attenuates_high_frequency_tone():
    sr = 4000
    high_tone = _tone(1500.0, 2.0, sr)
    out = lowpass_filter(high_tone, sr, cutoff_hz=300.0)
    assert _energy(out) < _energy(high_tone) * 0.1


def test_lowpass_preserves_low_frequency_tone():
    sr = 4000
    low_tone = _tone(100.0, 2.0, sr)
    out = lowpass_filter(low_tone, sr, cutoff_hz=300.0)
    assert _energy(out) > _energy(low_tone) * 0.5


def test_highpass_attenuates_low_frequency_tone():
    sr = 4000
    low_tone = _tone(50.0, 2.0, sr)
    out = highpass_filter(low_tone, sr, cutoff_hz=400.0)
    assert _energy(out) < _energy(low_tone) * 0.1


def test_highpass_preserves_high_frequency_tone():
    sr = 4000
    high_tone = _tone(1000.0, 2.0, sr)
    out = highpass_filter(high_tone, sr, cutoff_hz=400.0)
    assert _energy(out) > _energy(high_tone) * 0.5


def test_bandpass_keeps_only_the_band():
    sr = 4000
    in_band = _tone(500.0, 2.0, sr)
    below_band = _tone(50.0, 2.0, sr)
    above_band = _tone(1800.0, 2.0, sr)

    out_in = bandpass_filter(in_band, sr, low_hz=200.0, high_hz=800.0)
    out_below = bandpass_filter(below_band, sr, low_hz=200.0, high_hz=800.0)
    out_above = bandpass_filter(above_band, sr, low_hz=200.0, high_hz=800.0)

    assert _energy(out_in) > _energy(in_band) * 0.5
    assert _energy(out_below) < _energy(below_band) * 0.1
    assert _energy(out_above) < _energy(above_band) * 0.1


@pytest.mark.parametrize("cutoff_hz", [0, -100.0, 2000.0, 2500.0])
def test_lowpass_rejects_cutoff_at_or_beyond_nyquist(cutoff_hz: float):
    sr = 4000  # nyquist = 2000
    audio = np.zeros(sr, dtype=np.float32)
    with pytest.raises(FilterError):
        lowpass_filter(audio, sr, cutoff_hz=cutoff_hz)


@pytest.mark.parametrize("cutoff_hz", [0, -100.0, 2000.0])
def test_highpass_rejects_invalid_cutoff(cutoff_hz: float):
    sr = 4000
    audio = np.zeros(sr, dtype=np.float32)
    with pytest.raises(FilterError):
        highpass_filter(audio, sr, cutoff_hz=cutoff_hz)


def test_bandpass_rejects_low_greater_than_high():
    sr = 4000
    audio = np.zeros(sr, dtype=np.float32)
    with pytest.raises(FilterError):
        bandpass_filter(audio, sr, low_hz=800.0, high_hz=200.0)


def test_bandpass_rejects_high_at_or_beyond_nyquist():
    sr = 4000
    audio = np.zeros(sr, dtype=np.float32)
    with pytest.raises(FilterError):
        bandpass_filter(audio, sr, low_hz=200.0, high_hz=2000.0)


def test_bandpass_rejects_low_at_or_below_zero():
    sr = 4000
    audio = np.zeros(sr, dtype=np.float32)
    with pytest.raises(FilterError):
        bandpass_filter(audio, sr, low_hz=0.0, high_hz=500.0)


def test_filter_output_same_length_as_input():
    sr = 4000
    audio = _tone(300.0, 2.0, sr)
    out = lowpass_filter(audio, sr, cutoff_hz=500.0)
    assert len(out) == len(audio)

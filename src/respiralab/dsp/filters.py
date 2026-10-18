"""D11: optional high/low/bandpass filters with sample-rate-aware cutoffs.

Filtering is always opt-in: callers who skip these functions get the raw,
unfiltered signal — the "unfiltered control" docs/04_EXPERIMENTS.md asks for
is simply the absence of a filter call, not a separate code path. Cutoffs
are validated against each signal's OWN Nyquist frequency (sample_rate / 2)
and rejected (never silently clamped) when out of range — this is what
prevents docs/03_DATA_AND_EVALUATION_CONTRACT.md's "do not globally impose
an 8kHz bandpass on 4kHz sources" mistake: a cutoff valid for a 44.1kHz
recording will correctly raise FilterError on a 4kHz one.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


class FilterError(ValueError):
    """Raised for invalid cutoffs (<=0, >= Nyquist, or low >= high for bandpass)."""


def _validate_cutoff(cutoff_hz: float, nyquist_hz: float, name: str) -> None:
    if not (0 < cutoff_hz < nyquist_hz):
        raise FilterError(
            f"{name}={cutoff_hz}Hz must be in (0, {nyquist_hz}Hz) — the signal's own Nyquist "
            f"frequency at this sample rate, not a fixed assumption"
        )


def lowpass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth lowpass; cutoff_hz must be < sample_rate/2."""
    nyquist = sample_rate / 2
    _validate_cutoff(cutoff_hz, nyquist, "cutoff_hz")
    sos = butter(order, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return sosfiltfilt(sos, audio)


def highpass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth highpass; cutoff_hz must be < sample_rate/2."""
    nyquist = sample_rate / 2
    _validate_cutoff(cutoff_hz, nyquist, "cutoff_hz")
    sos = butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    return sosfiltfilt(sos, audio)


def bandpass_filter(
    audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass; 0 < low_hz < high_hz < sample_rate/2."""
    nyquist = sample_rate / 2
    _validate_cutoff(low_hz, nyquist, "low_hz")
    _validate_cutoff(high_hz, nyquist, "high_hz")
    if not low_hz < high_hz:
        raise FilterError(f"low_hz={low_hz} must be < high_hz={high_hz}")
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio)

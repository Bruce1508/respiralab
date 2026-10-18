from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


class FilterError(ValueError):
    pass


def _validate_cutoff(cutoff_hz: float, nyquist_hz: float, name: str) -> None:
    if not 0 < cutoff_hz < nyquist_hz:
        raise FilterError(
            f"{name}={cutoff_hz}Hz must be in (0, {nyquist_hz}Hz) — the signal's own Nyquist frequency at this sample rate, not a fixed assumption"
        )


def lowpass_filter(
    audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 4
) -> np.ndarray:
    nyquist = sample_rate / 2
    _validate_cutoff(cutoff_hz, nyquist, "cutoff_hz")
    sos = butter(order, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return sosfiltfilt(sos, audio)


def highpass_filter(
    audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 4
) -> np.ndarray:
    nyquist = sample_rate / 2
    _validate_cutoff(cutoff_hz, nyquist, "cutoff_hz")
    sos = butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    return sosfiltfilt(sos, audio)


def bandpass_filter(
    audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float, order: int = 4
) -> np.ndarray:
    nyquist = sample_rate / 2
    _validate_cutoff(low_hz, nyquist, "low_hz")
    _validate_cutoff(high_hz, nyquist, "high_hz")
    if not low_hz < high_hz:
        raise FilterError(f"low_hz={low_hz} must be < high_hz={high_hz}")
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio)

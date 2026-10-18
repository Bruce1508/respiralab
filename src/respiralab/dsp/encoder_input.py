"""D46: 16 kHz common-band encoder input path (ADR-0001 D2).

v2-only. Resamples to a pretrained-encoder-friendly rate, then low-passes to
the v1 upper edge (1800 Hz) so content above it -- present only on 10k/44.1k
devices -- cannot become a device shortcut (docs/07_SOURCE_VERIFICATION.md
item 4). v1's DSP path (dsp/resample.py, dsp/filters.py, models/dataset.py)
is untouched.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from respiralab.dsp.filters import lowpass_filter
from respiralab.dsp.resample import resample_audio

ENCODER_SAMPLE_RATE = 16000
COMMON_BAND_CUTOFF_HZ = 1800.0
TARGET_DURATION_S = 6.0  # matches v1's CNN cycle-length convention (models/dataset.py)
PadMode = Literal["zero", "repeat"]


def pad_or_crop(audio: np.ndarray, target_n_samples: int, mode: PadMode = "zero") -> np.ndarray:
    n = len(audio)
    if n >= target_n_samples:
        return audio[:target_n_samples]
    if mode == "zero":
        out = np.zeros(target_n_samples, dtype=audio.dtype)
        out[:n] = audio
        return out
    if mode == "repeat":
        if n == 0:
            raise ValueError("cannot repeat-pad an empty audio array")
        reps = -(-target_n_samples // n)  # ceil division
        return np.tile(audio, reps)[:target_n_samples]
    raise ValueError(f"unknown pad mode: {mode!r}")


def prepare_encoder_input(
    audio: np.ndarray,
    orig_sr: int,
    pad_mode: PadMode = "zero",
    target_sample_rate: int = ENCODER_SAMPLE_RATE,
    cutoff_hz: float = COMMON_BAND_CUTOFF_HZ,
    target_duration_s: float = TARGET_DURATION_S,
) -> np.ndarray:
    """Resample to target_sample_rate, low-pass to cutoff_hz, then fixed-length pad/crop."""
    resampled = resample_audio(audio, orig_sr, target_sample_rate)
    filtered = lowpass_filter(resampled, target_sample_rate, cutoff_hz)
    target_n_samples = int(target_sample_rate * target_duration_s)
    return pad_or_crop(filtered, target_n_samples, mode=pad_mode)

"""D09: sample-rate-safe resampling with anti-aliasing.

The dataset has 3 native sample rates (44100/10000/4000 Hz). Resampling uses
scipy's polyphase resampler (`resample_poly`), which applies a correctly
scaled anti-alias FIR low-pass filter matched to the *new* Nyquist frequency
as part of the resampling itself — never a fixed cutoff that could exceed a
lower-rate source's own Nyquist (docs/03_DATA_AND_EVALUATION_CONTRACT.md:
"Do not globally impose 8kHz bandpass on 4kHz sources").
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly


class ResampleError(ValueError):
    """Raised for invalid (non-positive) sample rates."""


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from orig_sr to target_sr with built-in anti-aliasing.

    A no-op (returns audio unchanged) when orig_sr == target_sr. Output
    length follows scipy's own resample_poly formula: ceil(len(audio) * up / down)
    for the reduced up/down ratio.
    """
    if orig_sr <= 0:
        raise ResampleError(f"orig_sr must be > 0, got {orig_sr}")
    if target_sr <= 0:
        raise ResampleError(f"target_sr must be > 0, got {target_sr}")
    if orig_sr == target_sr:
        return audio

    g = math.gcd(target_sr, orig_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(audio, up, down)

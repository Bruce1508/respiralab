from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly


class ResampleError(ValueError):
    pass


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr <= 0:
        raise ResampleError(f"orig_sr must be > 0, got {orig_sr}")
    if target_sr <= 0:
        raise ResampleError(f"target_sr must be > 0, got {target_sr}")
    if orig_sr == target_sr:
        return audio
    g = math.gcd(target_sr, orig_sr)
    up, down = (target_sr // g, orig_sr // g)
    return resample_poly(audio, up, down)

"""D12: handcrafted spectral/cepstral feature extraction.

Produces a fixed-length, finite feature vector regardless of input audio
duration — mean/std aggregation over frames collapses the time axis, so a
0.2s cycle and a 16s cycle both yield the same shape. This module is
inherently fit-free: every feature here is a direct per-cycle measurement,
never a parameter learned from the dataset. Scaling (StandardScaler etc.) is
the caller's job and MUST be fit on train only — that's enforced where the
scaler is actually fit (D16), not here.
"""

from __future__ import annotations

import librosa
import numpy as np

_EPS = 1e-10
_N_MFCC = 13
# librosa's default n_bands=6 needs frequencies up to fmin*2**6=12800Hz, which
# exceeds the Nyquist of this dataset's lowest native rate (4000Hz -> 2000Hz
# Nyquist) and raises. n_bands=3 (top edge 1600Hz) stays safely under 2000Hz
# Nyquist for ALL 3 native rates (4000/10000/44100), keeping the feature
# vector's shape fixed and crash-free regardless of which recording's native
# rate (or a post-D09-resample rate) a caller passes in.
_N_CONTRAST_BANDS = 3

FEATURE_NAMES: list[str] = (
    [f"mfcc_mean_{i}" for i in range(_N_MFCC)]
    + [f"mfcc_std_{i}" for i in range(_N_MFCC)]
    + ["spectral_centroid_mean", "spectral_centroid_std"]
    + ["spectral_bandwidth_mean", "spectral_bandwidth_std"]
    + ["spectral_rolloff_mean", "spectral_rolloff_std"]
    + ["spectral_flatness_mean", "spectral_flatness_std"]
    + ["rms_mean", "rms_std"]
    + ["zcr_mean", "zcr_std"]
    + [f"spectral_contrast_mean_{i}" for i in range(_N_CONTRAST_BANDS + 1)]
    + [f"spectral_contrast_std_{i}" for i in range(_N_CONTRAST_BANDS + 1)]
)


def _mean_std(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row mean/std over the frame axis; finite even for a single frame."""
    return frames.mean(axis=1), frames.std(axis=1)


def extract_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract the fixed-length handcrafted feature vector (len == len(FEATURE_NAMES))."""
    audio = audio.astype(np.float32)
    n_fft = min(256, max(len(audio), 1))
    hop_length = max(1, n_fft // 4)

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=_N_MFCC, n_fft=n_fft, hop_length=hop_length
    )
    mfcc_mean, mfcc_std = _mean_std(mfcc)

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_bands=_N_CONTRAST_BANDS
    )

    parts = [mfcc_mean, mfcc_std]
    for frames in (centroid, bandwidth, rolloff, flatness, rms, zcr):
        m, s = _mean_std(frames)
        parts.extend([m, s])
    contrast_mean, contrast_std = _mean_std(contrast)
    parts.extend([contrast_mean, contrast_std])

    vec = np.concatenate(parts).astype(np.float64)
    # floor any degenerate NaN/Inf (e.g. flatness/contrast on pathological silence)
    # rather than let a single bad cycle poison downstream training silently —
    # replace with 0.0 and let callers see it was a fallback via isfinite checks
    # on the raw librosa output if they need that distinction; here the contract
    # is simply "always finite, fixed shape" (docs/04_EXPERIMENTS.md).
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

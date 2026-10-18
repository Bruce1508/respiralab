from __future__ import annotations

import librosa
import numpy as np

_EPS = 1e-10
_N_MFCC = 13
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
    return (frames.mean(axis=1), frames.std(axis=1))


def extract_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = audio.astype(np.float32)
    n_fft = min(256, max(len(audio), 1))
    hop_length = max(1, n_fft // 4)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=_N_MFCC, n_fft=n_fft, hop_length=hop_length
    )
    mfcc_mean, mfcc_std = _mean_std(mfcc)
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
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
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

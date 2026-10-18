"""D10: waveform/PSD/STFT/log-Mel diagnostic computation and plotting.

Every dB/log conversion here floors magnitude with a small epsilon before
taking log10, so silence (all-zero audio) never produces -inf or NaN —
these are diagnostic tools that must stay usable on the dataset's real
silent/near-silent cycles, not just clean tones.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.signal import stft as scipy_stft
from scipy.signal import welch

_DB_EPSILON = 1e-10


def compute_psd(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density via Welch's method. Returns (freqs, psd)."""
    nperseg = min(len(audio), 1024)
    freqs, psd = welch(audio, fs=sample_rate, nperseg=nperseg)
    return freqs, psd


def compute_stft(
    audio: np.ndarray, sample_rate: int, n_fft: int = 256, hop_length: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier transform magnitude in dB. Returns (freqs, times, magnitude_db)."""
    nperseg = min(n_fft, len(audio))
    noverlap = max(0, nperseg - hop_length)
    freqs, times, Zxx = scipy_stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    magnitude_db = 20 * np.log10(np.abs(Zxx) + _DB_EPSILON)
    return freqs, times, magnitude_db


def compute_log_mel(
    audio: np.ndarray, sample_rate: int, n_mels: int = 64, n_fft: int = 256, hop_length: int = 64
) -> np.ndarray:
    """Log-Mel spectrogram (n_mels, n_frames), floored to avoid -inf on silence."""
    n_fft_eff = min(n_fft, max(len(audio), 1))
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft_eff,
        hop_length=hop_length,
    )
    return librosa.power_to_db(mel, ref=1.0, amin=_DB_EPSILON)


def save_diagnostic_figure(audio: np.ndarray, sample_rate: int, label: str, out_path: Path) -> None:
    """4-panel diagnostic figure: waveform, PSD, STFT spectrogram, log-Mel spectrogram."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs_psd, psd = compute_psd(audio, sample_rate)
    freqs_stft, times_stft, magnitude_db = compute_stft(audio, sample_rate)
    log_mel = compute_log_mel(audio, sample_rate)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Diagnostic plots — label: {label}")

    t = np.arange(len(audio)) / sample_rate
    axes[0, 0].plot(t, audio)
    axes[0, 0].set_title("Waveform")
    axes[0, 0].set_xlabel("time (s)")

    axes[0, 1].semilogy(freqs_psd, psd + _DB_EPSILON)
    axes[0, 1].set_title("Power spectral density")
    axes[0, 1].set_xlabel("frequency (Hz)")

    axes[1, 0].pcolormesh(times_stft, freqs_stft, magnitude_db, shading="gouraud")
    axes[1, 0].set_title("STFT magnitude (dB)")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("frequency (Hz)")

    axes[1, 1].imshow(log_mel, aspect="auto", origin="lower")
    axes[1, 1].set_title("log-Mel spectrogram (dB)")
    axes[1, 1].set_xlabel("frame")
    axes[1, 1].set_ylabel("mel bin")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

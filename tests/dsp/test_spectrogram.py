"""D10 tests (RED first): PSD/STFT/log-Mel computation — correct shapes,
finite output (no silent -inf/NaN from log(0)), diagnostic figure saving.
"""

from pathlib import Path

import numpy as np

from respiralab.dsp.spectrogram import (
    compute_log_mel,
    compute_psd,
    compute_stft,
    save_diagnostic_figure,
)


def _tone(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def test_compute_psd_returns_freqs_up_to_nyquist():
    sr = 4000
    audio = _tone(500.0, 2.0, sr)
    freqs, psd = compute_psd(audio, sr)
    assert freqs.max() <= sr / 2
    assert len(freqs) == len(psd)
    assert np.isfinite(psd).all()


def test_compute_psd_peaks_near_tone_frequency():
    sr = 4000
    audio = _tone(500.0, 2.0, sr)
    freqs, psd = compute_psd(audio, sr)
    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - 500.0) < 20.0  # within one PSD bin or so


def test_compute_stft_shapes_are_consistent():
    sr = 4000
    audio = _tone(500.0, 2.0, sr)
    freqs, times, magnitude_db = compute_stft(audio, sr)
    assert magnitude_db.shape == (len(freqs), len(times))
    assert np.isfinite(magnitude_db).all()


def test_compute_stft_handles_silence_without_negative_infinity():
    sr = 4000
    audio = np.zeros(sr * 2, dtype=np.float32)
    _, _, magnitude_db = compute_stft(audio, sr)
    assert np.isfinite(magnitude_db).all()  # not -inf from log(0)


def test_compute_log_mel_shape_and_finite():
    sr = 4000
    audio = _tone(500.0, 2.0, sr)
    log_mel = compute_log_mel(audio, sr, n_mels=32)
    assert log_mel.shape[0] == 32
    assert log_mel.shape[1] > 0
    assert np.isfinite(log_mel).all()


def test_compute_log_mel_handles_silence_without_negative_infinity():
    sr = 4000
    audio = np.zeros(sr * 2, dtype=np.float32)
    log_mel = compute_log_mel(audio, sr, n_mels=32)
    assert np.isfinite(log_mel).all()


def test_compute_log_mel_handles_very_short_audio():
    sr = 4000
    audio = _tone(500.0, 0.05, sr)  # 50ms, shorter than a typical n_fft
    log_mel = compute_log_mel(audio, sr, n_mels=32)
    assert log_mel.shape[0] == 32
    assert np.isfinite(log_mel).all()


def test_save_diagnostic_figure_writes_a_file(tmp_path: Path):
    sr = 4000
    audio = _tone(500.0, 2.0, sr)
    out = tmp_path / "example.png"
    save_diagnostic_figure(audio, sr, label="crackle", out_path=out)
    assert out.exists()
    assert out.stat().st_size > 0

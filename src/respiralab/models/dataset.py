from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from respiralab.dsp.resample import resample_audio
from respiralab.dsp.segment import extract_cycle
from respiralab.dsp.spectrogram import compute_log_mel
from respiralab.evaluation.metrics import LABEL_ORDER

TARGET_SAMPLE_RATE = 4000
MAX_DURATION_S = 6.0
TARGET_N_SAMPLES = int(TARGET_SAMPLE_RATE * MAX_DURATION_S)
HOP_LENGTH = 64
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABEL_ORDER)}


def pad_or_crop(audio: np.ndarray, target_n_samples: int) -> tuple[np.ndarray, int]:
    n = len(audio)
    if n >= target_n_samples:
        return audio[:target_n_samples], target_n_samples
    padded = np.zeros(target_n_samples, dtype=audio.dtype)
    padded[:n] = audio
    return padded, n


def build_frame_mask(
    n_real_samples: int, n_frames: int, hop_length: int = HOP_LENGTH
) -> np.ndarray:
    n_real_frames = min(n_frames, int(np.ceil(n_real_samples / hop_length)))
    mask = np.zeros(n_frames, dtype=bool)
    mask[:n_real_frames] = True
    return mask


class LogMelCycleDataset(Dataset):
    def __init__(
        self,
        cycles_df: pd.DataFrame,
        audio_dir: Path,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        target_n_samples: int = TARGET_N_SAMPLES,
    ) -> None:
        self.cycles_df = cycles_df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.target_sample_rate = target_sample_rate
        self.target_n_samples = target_n_samples
        self._audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.cycles_df)

    def _load_record_audio(self, record_id: str) -> tuple[np.ndarray, int]:
        if record_id not in self._audio_cache:
            audio, sample_rate = sf.read(str(self.audio_dir / f"{record_id}.wav"))
            self._audio_cache[record_id] = (audio, sample_rate)
        return self._audio_cache[record_id]

    def __getitem__(self, idx: int) -> dict:
        row = self.cycles_df.iloc[idx]
        audio, native_sr = self._load_record_audio(row["record_id"])
        cycle_audio = extract_cycle(audio, native_sr, row["start_s"], row["end_s"])
        resampled = resample_audio(cycle_audio, native_sr, self.target_sample_rate)
        fixed_audio, n_real_samples = pad_or_crop(resampled, self.target_n_samples)
        log_mel = compute_log_mel(fixed_audio, self.target_sample_rate)
        mask = build_frame_mask(n_real_samples, log_mel.shape[1])
        return {
            "log_mel": torch.from_numpy(log_mel).float(),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(LABEL_TO_INDEX[row["label4"]], dtype=torch.long),
            "cycle_id": row["cycle_id"],
        }

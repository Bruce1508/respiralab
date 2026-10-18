from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SegmentError(ValueError):
    pass


def extract_cycle(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    if start_s < 0:
        raise SegmentError(f"start_s must be >= 0, got {start_s}")
    if not start_s < end_s:
        raise SegmentError(f"start_s must be < end_s, got start={start_s} end={end_s}")
    start_sample = round(start_s * sample_rate)
    end_sample = round(end_s * sample_rate)
    if end_sample > len(audio):
        raise SegmentError(
            f"end_s={end_s} (sample {end_sample}) exceeds audio length ({len(audio)} samples at {sample_rate}Hz)"
        )
    if start_sample >= end_sample:
        raise SegmentError(
            f"rounded window is empty: start_sample={start_sample} end_sample={end_sample}"
        )
    return audio[start_sample:end_sample]


@dataclass(frozen=True)
class ExtractedCycle:
    cycle_id: str
    label4: str
    audio: np.ndarray


def extract_cycles_from_record(
    audio: np.ndarray, sample_rate: int, cycles: list[dict]
) -> list[ExtractedCycle]:
    return [
        ExtractedCycle(
            cycle_id=c["cycle_id"],
            label4=c["label4"],
            audio=extract_cycle(audio, sample_rate, c["start_s"], c["end_s"]),
        )
        for c in cycles
    ]

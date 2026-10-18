"""D08: deterministic cycle extraction respecting annotation bounds.

Crops the exact [start_s, end_s) window an annotation specifies — never a
fixed-size window — so no event-containing signal is silently discarded
(docs/04_EXPERIMENTS.md: "test whether crop discards wheezes or crackles").
Bounds validation here is deliberately strict and duplicates none of D04's
annotation-parsing validation: by the time a Cycle reaches this module its
start_s/end_s have already passed `annotation.parse_annotation_file`'s
checks against the recording's own duration; this module's own bounds check
exists for direct callers that didn't go through that path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SegmentError(ValueError):
    """Raised when start_s/end_s can't be cropped from the given audio array."""


def extract_cycle(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    """Crop audio[start_s:end_s) at sample_rate into a new array.

    Sample boundaries are rounded to the nearest sample (not truncated), so a
    cycle ending exactly at the recording's last sample is never shortened.
    Raises SegmentError (not silently clipped) if start >= end or the window
    falls outside the audio's actual length.
    """
    if start_s < 0:
        raise SegmentError(f"start_s must be >= 0, got {start_s}")
    if not start_s < end_s:
        raise SegmentError(f"start_s must be < end_s, got start={start_s} end={end_s}")

    start_sample = round(start_s * sample_rate)
    end_sample = round(end_s * sample_rate)

    if end_sample > len(audio):
        raise SegmentError(
            f"end_s={end_s} (sample {end_sample}) exceeds audio length "
            f"({len(audio)} samples at {sample_rate}Hz)"
        )
    if start_sample >= end_sample:
        raise SegmentError(f"rounded window is empty: start_sample={start_sample} end_sample={end_sample}")

    return audio[start_sample:end_sample]


@dataclass(frozen=True)
class ExtractedCycle:
    cycle_id: str
    label4: str
    audio: np.ndarray


def extract_cycles_from_record(
    audio: np.ndarray, sample_rate: int, cycles: list[dict]
) -> list[ExtractedCycle]:
    """Crop every cycle of one recording, keeping each segment linked to its cycle_id/label4.

    `cycles` is a list of dicts with at least cycle_id/start_s/end_s/label4
    (e.g. rows from manifests/cycles.csv filtered to one record_id). Every
    output is nonempty and bounded by construction (extract_cycle raises
    rather than returning an empty/truncated segment).
    """
    return [
        ExtractedCycle(
            cycle_id=c["cycle_id"],
            label4=c["label4"],
            audio=extract_cycle(audio, sample_rate, c["start_s"], c["end_s"]),
        )
        for c in cycles
    ]

"""D08 tests (RED first): deterministic cycle extraction — bounded, nonempty,
metadata-linked, never silently drops event-containing signal.
"""

import numpy as np
import pytest

from respiralab.dsp.segment import SegmentError, extract_cycle, extract_cycles_from_record


def test_extract_cycle_returns_exact_sample_range():
    sr = 4000
    audio = np.arange(sr * 4, dtype=np.float32)  # 4s of audio, value == sample index
    segment = extract_cycle(audio, sr, start_s=1.0, end_s=2.0)
    assert len(segment) == sr  # exactly 1 second
    assert segment[0] == 4000  # starts at sample 4000 (1.0s * 4000Hz)
    assert segment[-1] == 7999


def test_extract_cycle_is_nonempty_for_valid_bounds():
    sr = 4000
    audio = np.zeros(sr * 2, dtype=np.float32)
    segment = extract_cycle(audio, sr, start_s=0.036, end_s=0.579)
    assert len(segment) > 0


def test_extract_cycle_rejects_start_after_end():
    sr = 4000
    audio = np.zeros(sr * 2, dtype=np.float32)
    with pytest.raises(SegmentError):
        extract_cycle(audio, sr, start_s=1.0, end_s=0.5)


def test_extract_cycle_rejects_end_beyond_audio_length():
    sr = 4000
    audio = np.zeros(sr * 2, dtype=np.float32)  # 2s
    with pytest.raises(SegmentError):
        extract_cycle(audio, sr, start_s=0.0, end_s=5.0)  # way past 2s


def test_extract_cycle_never_silently_truncates_a_valid_window():
    # a cycle whose end lands exactly on the last sample must not be shortened
    sr = 4000
    audio = np.zeros(sr * 2, dtype=np.float32)
    segment = extract_cycle(audio, sr, start_s=0.0, end_s=2.0)
    assert len(segment) == sr * 2


def test_extract_cycles_from_record_links_metadata():
    sr = 4000
    audio = np.arange(sr * 4, dtype=np.float32)
    cycles = [
        {"cycle_id": "101_1b1_Al_sc_Meditron_c000", "start_s": 0.0, "end_s": 1.0, "label4": "normal"},
        {"cycle_id": "101_1b1_Al_sc_Meditron_c001", "start_s": 1.0, "end_s": 2.0, "label4": "crackle"},
    ]
    results = extract_cycles_from_record(audio, sr, cycles)

    assert len(results) == 2
    assert results[0].cycle_id == "101_1b1_Al_sc_Meditron_c000"
    assert results[0].label4 == "normal"
    assert len(results[0].audio) == sr
    assert results[1].cycle_id == "101_1b1_Al_sc_Meditron_c001"
    assert len(results[1].audio) > 0


def test_extract_cycles_from_record_all_outputs_bounded_and_nonempty():
    sr = 4000
    audio = np.zeros(sr * 10, dtype=np.float32)
    cycles = [
        {"cycle_id": f"r_c{i:03d}", "start_s": float(i), "end_s": float(i + 1), "label4": "normal"}
        for i in range(9)
    ]
    results = extract_cycles_from_record(audio, sr, cycles)
    assert len(results) == len(cycles)
    assert all(len(r.audio) > 0 for r in results)
    assert all(len(r.audio) <= len(audio) for r in results)

from __future__ import annotations

import pytest

from respiralab.data.hf_lung_labels import assign_phase_labels, parse_label_file


def test_parse_label_file_reads_type_start_end_lines():
    text = "I 00:00:03.638 00:00:04.458\nE 00:00:09.749 00:00:10.435\n"
    events = parse_label_file(text)
    assert len(events) == 2
    assert events[0].type == "I"
    assert events[0].start_s == pytest.approx(3.638)
    assert events[0].end_s == pytest.approx(4.458)
    assert events[1].type == "E"


def test_parse_label_file_skips_blank_lines():
    events = parse_label_file("I 00:00:00.000 00:00:01.000\n\n\n")
    assert len(events) == 1


def test_parse_label_file_raises_on_malformed_line():
    with pytest.raises(ValueError, match="malformed label line"):
        parse_label_file("I 00:00:00.000\n")  # missing end timestamp


def test_phase_with_no_overlap_is_normal():
    events = parse_label_file("I 00:00:00.000 00:00:02.000\n")
    labels = assign_phase_labels(events)
    assert len(labels) == 1
    assert labels[0].label4 == "normal"
    assert labels[0].phase_type == "I"


def test_phase_with_only_crackle_overlap_is_crackle():
    events = parse_label_file("I 00:00:00.000 00:00:02.000\nD 00:00:00.500 00:00:01.000\n")
    assert assign_phase_labels(events)[0].label4 == "crackle"


def test_phase_with_only_wheeze_overlap_is_wheeze():
    events = parse_label_file("I 00:00:00.000 00:00:02.000\nWheeze 00:00:00.500 00:00:01.000\n")
    assert assign_phase_labels(events)[0].label4 == "wheeze"


def test_phase_with_crackle_and_wheeze_overlap_is_both():
    events = parse_label_file(
        "I 00:00:00.000 00:00:02.000\nD 00:00:00.200 00:00:00.800\nWheeze 00:00:01.000 00:00:01.500\n"
    )
    assert assign_phase_labels(events)[0].label4 == "both"


def test_phase_with_only_rhonchi_or_stridor_overlap_is_excluded():
    events = parse_label_file("I 00:00:00.000 00:00:02.000\nRhonchi 00:00:00.500 00:00:01.000\n")
    assert assign_phase_labels(events)[0].label4 is None

    events = parse_label_file("E 00:00:00.000 00:00:02.000\nStridor 00:00:00.500 00:00:01.000\n")
    assert assign_phase_labels(events)[0].label4 is None


def test_crackle_plus_rhonchi_overlap_is_still_crackle_not_excluded():
    """Rhonchi never blocks a D/Wheeze verdict -- per the 2026-10-21 ADR-0002 clarification."""
    events = parse_label_file(
        "I 00:00:00.000 00:00:02.000\nD 00:00:00.200 00:00:00.800\nRhonchi 00:00:01.000 00:00:01.500\n"
    )
    assert assign_phase_labels(events)[0].label4 == "crackle"


def test_wheeze_plus_stridor_overlap_is_still_wheeze_not_excluded():
    events = parse_label_file(
        "E 00:00:00.000 00:00:02.000\nWheeze 00:00:00.200 00:00:00.800\nStridor 00:00:01.000 00:00:01.500\n"
    )
    assert assign_phase_labels(events)[0].label4 == "wheeze"


def test_non_overlapping_adventitious_event_does_not_count():
    """A D event entirely outside the phase's span must not be assigned to it."""
    events = parse_label_file("I 00:00:00.000 00:00:02.000\nD 00:00:05.000 00:00:06.000\n")
    assert assign_phase_labels(events)[0].label4 == "normal"


def test_multiple_phases_each_get_their_own_label():
    events = parse_label_file(
        "I 00:00:00.000 00:00:02.000\n"
        "D 00:00:00.500 00:00:01.000\n"
        "E 00:00:03.000 00:00:05.000\n"
        "Wheeze 00:00:03.500 00:00:04.000\n"
    )
    labels = assign_phase_labels(events)
    assert len(labels) == 2
    assert labels[0].label4 == "crackle"
    assert labels[1].label4 == "wheeze"

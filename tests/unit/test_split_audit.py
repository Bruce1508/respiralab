from pathlib import Path

import pytest

from respiralab.data.split_audit import audit_official_split, build_report, parse_official_split


def test_parse_official_split_reads_tab_separated(tmp_path: Path):
    f = tmp_path / "split.txt"
    f.write_text("101_1b1_Al_sc_Meditron\ttest\n103_2b2_Ar_mc_LittC2SE\ttrain\n")
    assignments = parse_official_split(f)
    assert assignments == {"101_1b1_Al_sc_Meditron": "test", "103_2b2_Ar_mc_LittC2SE": "train"}


def test_parse_official_split_rejects_bad_split_value(tmp_path: Path):
    f = tmp_path / "split.txt"
    f.write_text("101_1b1_Al_sc_Meditron\tvalidation\n")
    with pytest.raises(ValueError):
        parse_official_split(f)


def test_parse_official_split_rejects_malformed_row(tmp_path: Path):
    f = tmp_path / "split.txt"
    f.write_text("101_1b1_Al_sc_Meditron\n")
    with pytest.raises(ValueError):
        parse_official_split(f)


def test_audit_detects_no_overlap_when_patients_disjoint():
    assignments = {"101_1b1_Al_sc_Meditron": "train", "102_1b1_Ar_sc_Meditron": "test"}
    result = audit_official_split(assignments)
    assert result.overlapping_patients == frozenset()
    assert result.train_patients == frozenset({"101"})
    assert result.test_patients == frozenset({"102"})


def test_audit_detects_overlap_when_same_patient_in_both():
    assignments = {"218_1b1_Al_sc_Meditron": "test", "218_1b1_Ar_sc_Meditron": "train"}
    result = audit_official_split(assignments)
    assert result.overlapping_patients == frozenset({"218"})


def test_report_states_not_disjoint_when_overlap_found():
    assignments = {"156_2b3_Al_mc_AKGC417L": "test", "156_2b3_Ar_mc_AKGC417L": "train"}
    result = audit_official_split(assignments)
    report = build_report(result)
    assert "NOT patient-disjoint" in report
    assert "156" in report


def test_report_states_disjoint_when_no_overlap():
    assignments = {"101_1b1_Al_sc_Meditron": "train", "102_1b1_Ar_sc_Meditron": "test"}
    result = audit_official_split(assignments)
    report = build_report(result)
    assert "**Patient-disjoint**" in report
    assert "no overlap found" in report

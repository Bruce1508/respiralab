from pathlib import Path

import pytest

from respiralab.data.annotation import (
    AnnotationParseError,
    build_cycle_manifest,
    label4_from_bits,
    load_duration_lookup,
    parse_annotation_file,
    write_csv,
)


@pytest.mark.parametrize(
    "crackle,wheeze,expected",
    [(0, 0, "normal"), (1, 0, "crackle"), (0, 1, "wheeze"), (1, 1, "both")],
)
def test_label_map_four_combinations(crackle: int, wheeze: int, expected: str):
    assert label4_from_bits(crackle, wheeze) == expected


def test_label_map_rejects_out_of_range_bits():
    with pytest.raises(AnnotationParseError):
        label4_from_bits(2, 0)


def test_parses_real_shaped_file(tmp_path: Path):
    f = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    f.write_text("0.036 0.579 0 0\n0.579 2.45 1 0\n2.45 3.893 0 1\n3.893 5.793 1 1\n")
    cycles = parse_annotation_file(f)
    assert len(cycles) == 4
    assert [c.label4 for c in cycles] == ["normal", "crackle", "wheeze", "both"]
    assert all(c.record_id == "101_1b1_Al_sc_Meditron" for c in cycles)
    assert all(c.patient_id == "101" for c in cycles)
    assert cycles[0].cycle_id == "101_1b1_Al_sc_Meditron_c000"


def test_rejects_malformed_column_count(tmp_path: Path):
    f = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    f.write_text("0.0 1.0 0\n")
    with pytest.raises(AnnotationParseError):
        parse_annotation_file(f)


def test_rejects_non_numeric_time(tmp_path: Path):
    f = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    f.write_text("abc 1.0 0 0\n")
    with pytest.raises(AnnotationParseError):
        parse_annotation_file(f)


def test_rejects_invalid_bit_value(tmp_path: Path):
    f = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    f.write_text("0.0 1.0 2 0\n")
    with pytest.raises(AnnotationParseError):
        parse_annotation_file(f)


def test_rejects_start_after_end(tmp_path: Path):
    f = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    f.write_text("2.0 1.0 0 0\n")
    with pytest.raises(AnnotationParseError):
        parse_annotation_file(f)


def test_rejects_negative_start(tmp_path: Path):
    f = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    f.write_text("-0.5 1.0 0 0\n")
    with pytest.raises(AnnotationParseError):
        parse_annotation_file(f)


def test_rejects_end_beyond_audio_duration_even_with_tolerance(tmp_path: Path):
    f = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    f.write_text("0.0 5.0 0 0\n")
    with pytest.raises(AnnotationParseError):
        parse_annotation_file(f, audio_duration_s=4.0, tolerance_s=0.001)


def test_accepts_end_within_tolerance_of_audio_duration(tmp_path: Path):
    f = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    f.write_text("0.0 4.0005 0 0\n")
    cycles = parse_annotation_file(f, audio_duration_s=4.0, tolerance_s=0.001)
    assert len(cycles) == 1


def test_unparseable_filename_raises_not_silently_skips(tmp_path: Path):
    f = tmp_path / "not_an_icbhi_name.txt"
    f.write_text("0.0 1.0 0 0\n")
    with pytest.raises(AnnotationParseError):
        parse_annotation_file(f)


def test_build_cycle_manifest_logs_rejection_without_raising(tmp_path: Path):
    good = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    good.write_text("0.0 1.0 0 0\n")
    bad = tmp_path / "102_1b1_Ar_sc_Meditron.txt"
    bad.write_text("2.0 1.0 0 0\n")
    cycles, rejections = build_cycle_manifest(tmp_path)
    assert len(cycles) == 1
    assert len(rejections) == 1
    assert rejections[0].filename == "102_1b1_Ar_sc_Meditron.txt"


def test_build_cycle_manifest_enforces_duration_lookup(tmp_path: Path):
    f = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    f.write_text("0.0 10.0 0 0\n")
    cycles, rejections = build_cycle_manifest(
        tmp_path, duration_lookup={"101_1b1_Al_sc_Meditron": (4.0, 4000)}
    )
    assert cycles == []
    assert len(rejections) == 1


def test_load_duration_lookup_reads_dataset_inventory_csv(tmp_path: Path):
    csv_path = tmp_path / "dataset_inventory.csv"
    csv_path.write_text(
        "record_id,patient_id,recording_index,chest_site,mode,device,sample_rate,channels,duration_s,size_bytes,sha256,source_filename\n101_1b1_Al_sc_Meditron,101,1b1,Al,sc,Meditron,4000,1,20.5,100,abc,101_1b1_Al_sc_Meditron.wav\n"
    )
    lookup = load_duration_lookup(csv_path)
    assert lookup == {"101_1b1_Al_sc_Meditron": (20.5, 4000)}


def test_write_csv_matches_dataclass_fields(tmp_path: Path):
    f = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    f.write_text("0.0 1.0 1 1\n")
    cycles = parse_annotation_file(f)
    out = tmp_path / "cycles.csv"
    write_csv(cycles, out)
    import csv as csv_mod

    with out.open() as fh:
        reader = csv_mod.DictReader(fh)
        assert reader.fieldnames == [
            "cycle_id",
            "record_id",
            "patient_id",
            "start_s",
            "end_s",
            "crackle",
            "wheeze",
            "label4",
        ]
        row = next(reader)
        assert row["label4"] == "both"

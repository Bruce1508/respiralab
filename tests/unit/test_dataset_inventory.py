import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from respiralab.data.dataset_inventory import build_dataset_inventory, write_csv


def _write_wav(path: Path, duration_s: float, sample_rate: int) -> None:
    samples = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    sf.write(str(path), samples, sample_rate)


def test_build_dataset_inventory_reads_header_and_hash(tmp_path: Path):
    _write_wav(tmp_path / "101_1b1_Al_sc_Meditron.wav", duration_s=2.0, sample_rate=4000)
    rows, rejections = build_dataset_inventory(tmp_path)
    assert rejections == []
    assert len(rows) == 1
    row = rows[0]
    assert row.patient_id == "101"
    assert row.chest_site == "Al"
    assert row.device == "Meditron"
    assert row.sample_rate == 4000
    assert row.channels == 1
    assert abs(row.duration_s - 2.0) < 0.001
    assert len(row.sha256) == 64


def test_unparseable_filename_is_rejected_not_skipped(tmp_path: Path):
    _write_wav(tmp_path / "not_an_icbhi_name.wav", duration_s=1.0, sample_rate=4000)
    rows, rejections = build_dataset_inventory(tmp_path)
    assert rows == []
    assert len(rejections) == 1
    assert rejections[0].filename == "not_an_icbhi_name.wav"


def test_corrupt_wav_is_rejected_not_skipped(tmp_path: Path):
    bad = tmp_path / "101_1b1_Al_sc_Meditron.wav"
    bad.write_bytes(b"not actually a wav file")
    rows, rejections = build_dataset_inventory(tmp_path)
    assert rows == []
    assert len(rejections) == 1
    assert "unreadable WAV header" in rejections[0].reason


def test_mixed_valid_and_rejected_both_logged_by_filename(tmp_path: Path):
    _write_wav(tmp_path / "101_1b1_Al_sc_Meditron.wav", duration_s=1.0, sample_rate=44100)
    (tmp_path / "bad_name.wav").write_bytes(b"garbage")
    rows, rejections = build_dataset_inventory(tmp_path)
    assert len(rows) == 1
    assert len(rejections) == 1
    assert rejections[0].filename == "bad_name.wav"


def test_write_csv_matches_dataclass_fields(tmp_path: Path):
    _write_wav(tmp_path / "101_1b1_Al_sc_Meditron.wav", duration_s=1.0, sample_rate=4000)
    rows, _ = build_dataset_inventory(tmp_path)
    out = tmp_path / "manifest.csv"
    write_csv(rows, out)
    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "record_id",
            "patient_id",
            "recording_index",
            "chest_site",
            "mode",
            "device",
            "sample_rate",
            "channels",
            "duration_s",
            "size_bytes",
            "sha256",
            "source_filename",
        ]
        assert len(list(reader)) == 1

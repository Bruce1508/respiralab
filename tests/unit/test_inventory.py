"""D02 unit tests: sha256 correctness, size/readability reporting, CSV shape, failure handling."""

import csv
import hashlib
from pathlib import Path

from respiralab.data.inventory import build_inventory, sha256_of_file, write_inventory_csv


def test_sha256_matches_known_digest(tmp_path: Path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"hello respiralab")
    expected = hashlib.sha256(b"hello respiralab").hexdigest()
    assert sha256_of_file(f) == expected


def test_build_inventory_reports_size_and_hash(tmp_path: Path):
    (tmp_path / "101_1b1_Al_sc.wav").write_bytes(b"\x00" * 100)
    (tmp_path / "101_1b1_Al_sc.txt").write_text("0.0 0.9 0 0\n")
    (tmp_path / "ignored.csv").write_text("not scanned by default extension filter")

    rows = build_inventory(tmp_path, extensions=(".wav", ".txt"))

    assert {r.relative_path for r in rows} == {"101_1b1_Al_sc.wav", "101_1b1_Al_sc.txt"}
    wav_row = next(r for r in rows if r.relative_path.endswith(".wav"))
    assert wav_row.size_bytes == 100
    assert wav_row.readable is True
    assert wav_row.error == ""


def test_build_inventory_records_failure_without_raising(tmp_path: Path):
    good = tmp_path / "good.wav"
    good.write_bytes(b"data")
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"data")
    broken.chmod(0o000)

    try:
        rows = build_inventory(tmp_path, extensions=(".wav",))
    finally:
        broken.chmod(0o644)  # restore so tmp_path cleanup doesn't fail

    by_name = {r.relative_path: r for r in rows}
    assert by_name["good.wav"].readable is True
    assert by_name["broken.wav"].readable is False
    assert by_name["broken.wav"].error != ""
    assert by_name["broken.wav"].size_bytes is None


def test_write_inventory_csv_has_expected_columns(tmp_path: Path):
    (tmp_path / "x.wav").write_bytes(b"abc")
    rows = build_inventory(tmp_path, extensions=(".wav",))
    out = tmp_path / "manifest.csv"
    write_inventory_csv(rows, out)

    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["relative_path", "size_bytes", "sha256", "readable", "error"]
        rows_read = list(reader)
    assert len(rows_read) == 1
    assert rows_read[0]["relative_path"] == "x.wav"

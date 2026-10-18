"""D03: WAV header audit — duration, sample rate, channels — plus filename
decode and checksum, with every rejection logged by filename (never silently
dropped). Builds on D02's `inventory.sha256_of_file` for hashing and
`filenames.parse_icbhi_filename` for the naming convention; reads only WAV
*headers* via `soundfile.info` (no full audio decode) to stay fast and keep
this a metadata audit, not a DSP step.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import soundfile as sf

from respiralab.data.filenames import FilenameParseError, parse_icbhi_filename
from respiralab.data.inventory import sha256_of_file

EXPECTED_RECORDINGS = 920
EXPECTED_PATIENTS = 126


@dataclass
class DatasetInventoryRow:
    record_id: str
    patient_id: str
    recording_index: str
    chest_site: str
    mode: str
    device: str
    sample_rate: int
    channels: int
    duration_s: float
    size_bytes: int
    sha256: str
    source_filename: str


@dataclass
class RejectionRow:
    filename: str
    reason: str


def build_dataset_inventory(root: Path) -> tuple[list[DatasetInventoryRow], list[RejectionRow]]:
    """Audit every .wav under root: parse filename, read header, hash content.

    Each file either produces exactly one DatasetInventoryRow or exactly one
    RejectionRow (filename-parse failure or unreadable/corrupt WAV header) —
    no file is silently skipped.
    """
    rows: list[DatasetInventoryRow] = []
    rejections: list[RejectionRow] = []
    for path in sorted(root.glob("*.wav")):
        try:
            parsed = parse_icbhi_filename(path.stem)
        except FilenameParseError as exc:
            rejections.append(RejectionRow(path.name, str(exc)))
            continue
        try:
            info = sf.info(str(path))
            digest = sha256_of_file(path)
            size = path.stat().st_size
        except (RuntimeError, OSError) as exc:
            rejections.append(RejectionRow(path.name, f"unreadable WAV header: {exc}"))
            continue
        rows.append(
            DatasetInventoryRow(
                record_id=parsed.record_id,
                patient_id=parsed.patient_id,
                recording_index=parsed.recording_index,
                chest_site=parsed.chest_site,
                mode=parsed.mode,
                device=parsed.device,
                sample_rate=info.samplerate,
                channels=info.channels,
                duration_s=round(info.duration, 6),
                size_bytes=size,
                sha256=digest,
                source_filename=path.name,
            )
        )
    return rows, rejections


def write_csv(dataclass_rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataclass_rows:
        out_path.write_text("")
        return
    fieldnames = [f.name for f in fields(dataclass_rows[0])]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataclass_rows:
            writer.writerow(vars(row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit ICBHI WAV headers (duration/sr/channels) + filenames + checksums; "
        "log every rejection by filename."
    )
    parser.add_argument("--dir", required=True, type=Path, help="Directory of .wav files to audit")
    parser.add_argument("--out", required=True, type=Path, help="Output dataset_inventory.csv path")
    parser.add_argument(
        "--rejections-out",
        type=Path,
        default=None,
        help="Output rejections CSV path (default: <out>_rejections.csv)",
    )
    args = parser.parse_args(argv)

    if not args.dir.is_dir():
        print(f"error: --dir {args.dir} is not a directory", file=sys.stderr)
        return 1

    rejections_out = args.rejections_out or args.out.with_name(
        args.out.stem + "_rejections" + args.out.suffix
    )
    rows, rejections = build_dataset_inventory(args.dir)
    write_csv(rows, args.out)
    write_csv(rejections, rejections_out)

    n_patients = len({r.patient_id for r in rows})
    print(f"audited {len(rows)} wav files ({n_patients} unique patients) under {args.dir}")
    print(f"wrote manifest: {args.out}")
    print(f"wrote rejections: {rejections_out} ({len(rejections)} entries)")
    if len(rows) != EXPECTED_RECORDINGS:
        print(
            f"NOTE: expected {EXPECTED_RECORDINGS} recordings, audited {len(rows)} — "
            "documented discrepancy, not auto-corrected.",
            file=sys.stderr,
        )
    if n_patients != EXPECTED_PATIENTS:
        print(
            f"NOTE: expected {EXPECTED_PATIENTS} patients, found {n_patients} — "
            "documented discrepancy, not auto-corrected.",
            file=sys.stderr,
        )
    if rejections:
        print(f"REJECTIONS ({len(rejections)}):", file=sys.stderr)
        for r in rejections:
            print(f"  {r.filename}: {r.reason}", file=sys.stderr)
    return 1 if rejections else 0


if __name__ == "__main__":
    raise SystemExit(main())

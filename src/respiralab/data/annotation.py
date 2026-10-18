from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, fields
from pathlib import Path

from respiralab.data.filenames import FilenameParseError, RejectionRow, parse_icbhi_filename

LABEL_MAP: dict[tuple[int, int], str] = {
    (0, 0): "normal",
    (1, 0): "crackle",
    (0, 1): "wheeze",
    (1, 1): "both",
}


class AnnotationParseError(ValueError):
    pass


@dataclass(frozen=True)
class Cycle:
    cycle_id: str
    record_id: str
    patient_id: str
    start_s: float
    end_s: float
    crackle: int
    wheeze: int
    label4: str


def label4_from_bits(crackle: int, wheeze: int) -> str:
    try:
        return LABEL_MAP[crackle, wheeze]
    except KeyError as exc:
        raise AnnotationParseError(
            f"crackle/wheeze must each be 0 or 1, got crackle={crackle!r} wheeze={wheeze!r}"
        ) from exc


def _parse_bit(token: str, field_name: str, context: str) -> int:
    try:
        value = int(float(token))
    except ValueError as exc:
        raise AnnotationParseError(f"{context}: {field_name} is not numeric: {token!r}") from exc
    if value not in (0, 1):
        raise AnnotationParseError(f"{context}: {field_name} must be 0 or 1, got {value}")
    return value


def parse_annotation_file(
    path: Path, *, audio_duration_s: float | None = None, tolerance_s: float = 0.0
) -> list[Cycle]:
    try:
        parsed_name = parse_icbhi_filename(path.stem)
    except FilenameParseError as exc:
        raise AnnotationParseError(f"{path.name}: {exc}") from exc
    cycles: list[Cycle] = []
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        context = f"{path.name}:{line_no}"
        parts = line.split()
        if len(parts) != 4:
            raise AnnotationParseError(
                f"{context}: expected 4 columns, got {len(parts)}: {raw_line!r}"
            )
        start_tok, end_tok, crackle_tok, wheeze_tok = parts
        try:
            start_s = float(start_tok)
            end_s = float(end_tok)
        except ValueError as exc:
            raise AnnotationParseError(
                f"{context}: start/end must be numeric: {raw_line!r}"
            ) from exc
        crackle = _parse_bit(crackle_tok, "crackle", context)
        wheeze = _parse_bit(wheeze_tok, "wheeze", context)
        if start_s < 0:
            raise AnnotationParseError(f"{context}: start_s must be >= 0, got {start_s}")
        if not start_s < end_s:
            raise AnnotationParseError(
                f"{context}: start_s must be < end_s, got start={start_s} end={end_s}"
            )
        if audio_duration_s is not None and end_s > audio_duration_s + tolerance_s:
            raise AnnotationParseError(
                f"{context}: end_s={end_s} exceeds audio_duration_s={audio_duration_s} (+tolerance_s={tolerance_s})"
            )
        cycles.append(
            Cycle(
                cycle_id=f"{parsed_name.record_id}_c{len(cycles):03d}",
                record_id=parsed_name.record_id,
                patient_id=parsed_name.patient_id,
                start_s=start_s,
                end_s=end_s,
                crackle=crackle,
                wheeze=wheeze,
                label4=label4_from_bits(crackle, wheeze),
            )
        )
    return cycles


def build_cycle_manifest(
    txt_dir: Path, duration_lookup: dict[str, tuple[float, int]] | None = None
) -> tuple[list[Cycle], list[RejectionRow]]:
    cycles: list[Cycle] = []
    rejections: list[RejectionRow] = []
    for path in sorted(txt_dir.glob("*.txt")):
        duration_s = tolerance_s = None
        if duration_lookup is not None and path.stem in duration_lookup:
            duration_s, sample_rate = duration_lookup[path.stem]
            tolerance_s = 1.0 / sample_rate
        try:
            cycles.extend(
                parse_annotation_file(
                    path, audio_duration_s=duration_s, tolerance_s=tolerance_s or 0.0
                )
            )
        except AnnotationParseError as exc:
            rejections.append(RejectionRow(path.name, str(exc)))
    return (cycles, rejections)


def load_duration_lookup(dataset_inventory_csv: Path) -> dict[str, tuple[float, int]]:
    with dataset_inventory_csv.open() as f:
        return {
            row["record_id"]: (float(row["duration_s"]), int(row["sample_rate"]))
            for row in csv.DictReader(f)
        }


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
        description="Parse ICBHI annotation .txt files into a cycle-level manifest with the 4-class label map; logs every rejection by filename."
    )
    parser.add_argument(
        "--dir", required=True, type=Path, help="Directory of .txt annotation files"
    )
    parser.add_argument("--out", required=True, type=Path, help="Output cycles.csv path")
    parser.add_argument(
        "--dataset-inventory",
        type=Path,
        default=None,
        help="D03's dataset_inventory.csv, to enforce end_s <= duration (+1 sample)",
    )
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
    duration_lookup = (
        load_duration_lookup(args.dataset_inventory) if args.dataset_inventory else None
    )
    rejections_out = args.rejections_out or args.out.with_name(
        args.out.stem + "_rejections" + args.out.suffix
    )
    cycles, rejections = build_cycle_manifest(args.dir, duration_lookup)
    write_csv(cycles, args.out)
    write_csv(rejections, rejections_out)
    from collections import Counter

    label_counts = Counter(c.label4 for c in cycles)
    print(f"parsed {len(cycles)} cycles from {args.dir}")
    print(f"label counts: {dict(label_counts)}")
    print(f"wrote manifest: {args.out}")
    print(f"wrote rejections: {rejections_out} ({len(rejections)} entries)")
    if rejections:
        print(f"REJECTIONS ({len(rejections)}):", file=sys.stderr)
        for r in rejections:
            print(f"  {r.filename}: {r.reason}", file=sys.stderr)
    return 1 if rejections else 0


if __name__ == "__main__":
    raise SystemExit(main())

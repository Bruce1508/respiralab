"""File-level checksum/inventory for the raw dataset directory.

Scope (D02): generic file inventory — size, sha256, readability — over
whatever is in `data/raw/`. It does NOT parse WAV headers or annotation
content (sample rate, duration, channels, cycle labels); that is D03's
`respiralab.data.annotation`/header-parsing work, which will consume this
module's CSV as a starting point. This module never reads raw audio bytes
into memory beyond what's needed to hash them, and never commits or prints
file content — only file metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB; bounds memory for arbitrarily large files


@dataclass
class InventoryRow:
    relative_path: str
    size_bytes: int | None
    sha256: str | None
    readable: bool
    error: str


def sha256_of_file(path: Path, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Stream a file through sha256 in fixed-size chunks (no full-file load)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path, extensions: tuple[str, ...] | None = None) -> Iterator[Path]:
    """Yield files under root, sorted for deterministic output, optionally filtered by extension."""
    candidates = sorted(p for p in root.rglob("*") if p.is_file())
    if extensions is None:
        return iter(candidates)
    lowered = tuple(e.lower() for e in extensions)
    return iter(p for p in candidates if p.suffix.lower() in lowered)


def build_inventory(root: Path, extensions: tuple[str, ...] | None = None) -> list[InventoryRow]:
    """Hash and size every matching file under root; record (not raise on) per-file failures."""
    rows: list[InventoryRow] = []
    for path in iter_files(root, extensions):
        rel = str(path.relative_to(root))
        try:
            size = path.stat().st_size
            digest = sha256_of_file(path)
            rows.append(InventoryRow(rel, size, digest, True, ""))
        except OSError as exc:
            rows.append(InventoryRow(rel, None, None, False, str(exc)))
    return rows


def write_inventory_csv(rows: list[InventoryRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(InventoryRow)]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(vars(row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Checksum/inventory raw dataset files (sizes + sha256 + failures). "
        "Never commits or prints file content."
    )
    parser.add_argument("--dir", required=True, type=Path, help="Directory to scan (e.g. data/raw)")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV manifest path")
    parser.add_argument(
        "--ext",
        default=".wav,.txt",
        help="Comma-separated extensions to include (default: .wav,.txt); empty string = all files",
    )
    args = parser.parse_args(argv)

    if not args.dir.is_dir():
        print(f"error: --dir {args.dir} is not a directory", file=sys.stderr)
        return 1

    extensions = tuple(e.strip() for e in args.ext.split(",") if e.strip()) or None
    rows = build_inventory(args.dir, extensions)
    write_inventory_csv(rows, args.out)

    failures = [r for r in rows if not r.readable]
    total_bytes = sum(r.size_bytes for r in rows if r.size_bytes is not None)
    print(f"scanned {len(rows)} files under {args.dir} ({total_bytes} bytes total)")
    print(f"wrote manifest: {args.out}")
    if failures:
        print(f"FAILURES ({len(failures)}):", file=sys.stderr)
        for r in failures:
            print(f"  {r.relative_path}: {r.error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

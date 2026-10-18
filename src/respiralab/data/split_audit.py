from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from respiralab.data.filenames import parse_icbhi_filename

REPORTED_OVERLAP_PATIENT_IDS = frozenset({"156", "218"})


@dataclass(frozen=True)
class SplitAuditResult:
    n_rows: int
    train_patients: frozenset[str]
    test_patients: frozenset[str]
    overlapping_patients: frozenset[str]


def parse_official_split(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) != 2:
            raise ValueError(
                f"{path.name}:{line_no}: expected 2 columns, got {len(parts)}: {raw_line!r}"
            )
        record_id, split = parts
        if split not in ("train", "test"):
            raise ValueError(
                f"{path.name}:{line_no}: split must be 'train' or 'test', got {split!r}"
            )
        assignments[record_id] = split
    return assignments


def audit_official_split(assignments: dict[str, str]) -> SplitAuditResult:
    train_patients: set[str] = set()
    test_patients: set[str] = set()
    for record_id, split in assignments.items():
        patient_id = parse_icbhi_filename(record_id).patient_id
        (train_patients if split == "train" else test_patients).add(patient_id)
    overlap = frozenset(train_patients & test_patients)
    return SplitAuditResult(
        n_rows=len(assignments),
        train_patients=frozenset(train_patients),
        test_patients=frozenset(test_patients),
        overlapping_patients=overlap,
    )


def build_report(result: SplitAuditResult) -> str:
    confirmed = REPORTED_OVERLAP_PATIENT_IDS & result.overlapping_patients
    unconfirmed = REPORTED_OVERLAP_PATIENT_IDS - result.overlapping_patients
    extra = result.overlapping_patients - REPORTED_OVERLAP_PATIENT_IDS
    overlap_table = (
        "\n".join(f"| {p} |" for p in sorted(result.overlapping_patients, key=int))
        if result.overlapping_patients
        else "_(none)_"
    )
    verdict = (
        "**NOT patient-disjoint.**"
        if result.overlapping_patients
        else "**Patient-disjoint** — no overlap found."
    )
    return f"# Official ICBHI split audit (D06)\n\nSource: `ICBHI_challenge_train_test.txt` (official page, downloaded directly;\n920 rows, one per recording — matches the full dataset).\n\n## Verdict\n{verdict} This project's primary evaluation protocol (D07) uses its own\nfrozen patient-disjoint split regardless of this result — the official split\nis reported here as a separate, audited comparison protocol only\n(docs/03_DATA_AND_EVALUATION_CONTRACT.md), never combined with primary-split\nnumbers in one leaderboard.\n\n## Counts\n- Rows parsed: {result.n_rows}\n- Unique train patients: {len(result.train_patients)}\n- Unique test patients: {len(result.test_patients)}\n- Patients appearing in **both** train and test: {len(result.overlapping_patients)}\n\n## Secondary-source claim check (patient IDs 156, 218)\n- Confirmed overlapping in the actual official file: {sorted(confirmed) or 'none'}\n- Claimed but NOT found overlapping here: {sorted(unconfirmed) or 'none'}\n- Additional overlapping patients beyond the two claimed: {sorted(extra, key=int) or 'none'}\n\n## All overlapping patient IDs\n| patient_id |\n|---|\n{overlap_table}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the official ICBHI split for patient overlap."
    )
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    args = parser.parse_args(argv)
    assignments = parse_official_split(args.split_file)
    result = audit_official_split(assignments)
    report = build_report(result)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report)
    print(f"rows: {result.n_rows}")
    print(f"overlapping patients: {sorted(result.overlapping_patients, key=int)}")
    print(f"wrote report: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

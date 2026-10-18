"""D05: EDA over D03's dataset_inventory.csv and D04's cycles.csv — class,
device, chest-site, and duration distributions — writing reports/data_audit.md
and bar/histogram plots to reports/figures/. Counts are cross-checked against
the official ICBHI totals (docs/03_DATA_AND_EVALUATION_CONTRACT.md); any
mismatch is enumerated in the report, never silently absorbed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EXPECTED_RECORDINGS = 920
EXPECTED_PATIENTS = 126
EXPECTED_CYCLES = 6898
EXPECTED_LABEL_COUNTS = {"normal": 3642, "crackle": 1864, "wheeze": 886, "both": 506}


@dataclass
class Discrepancy:
    dimension: str
    expected: object
    actual: object


def compute_discrepancies(
    n_recordings: int,
    n_patients: int,
    n_cycles: int,
    label_counts: dict[str, int],
) -> list[Discrepancy]:
    """Compare actual counts to the official totals; return every mismatch found."""
    discrepancies: list[Discrepancy] = []
    checks = [
        ("recordings", EXPECTED_RECORDINGS, n_recordings),
        ("patients", EXPECTED_PATIENTS, n_patients),
        ("cycles", EXPECTED_CYCLES, n_cycles),
    ]
    for label, expected_count in EXPECTED_LABEL_COUNTS.items():
        checks.append((f"label:{label}", expected_count, label_counts.get(label, 0)))
    for dimension, expected, actual in checks:
        if expected != actual:
            discrepancies.append(Discrepancy(dimension, expected, actual))
    return discrepancies


def _bar_plot(counts: Counter, title: str, xlabel: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: never requires a display
    import matplotlib.pyplot as plt

    items = sorted(counts.items(), key=lambda kv: kv[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(k) for k, _ in items], [v for _, v in items])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _histogram(values: pd.Series, title: str, xlabel: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def build_report(
    inventory_df: pd.DataFrame,
    cycles_df: pd.DataFrame,
    figures_dir: Path,
) -> tuple[str, list[Discrepancy]]:
    """Compute all summary stats, write the 4 plots, and return the markdown report text."""
    n_recordings = len(inventory_df)
    n_patients = inventory_df["patient_id"].nunique()
    n_cycles = len(cycles_df)
    label_counts = Counter(cycles_df["label4"])
    device_counts = Counter(inventory_df["device"])
    chest_site_counts = Counter(inventory_df["chest_site"])
    sample_rate_counts = Counter(inventory_df["sample_rate"])

    discrepancies = compute_discrepancies(n_recordings, n_patients, n_cycles, dict(label_counts))

    _bar_plot(label_counts, "Cycle label distribution", "label4", figures_dir / "labels.png")
    _bar_plot(device_counts, "Recording device distribution", "device", figures_dir / "devices.png")
    _bar_plot(
        chest_site_counts, "Chest site distribution", "chest_site", figures_dir / "chest_sites.png"
    )
    _histogram(
        inventory_df["duration_s"], "Recording duration distribution", "duration_s",
        figures_dir / "durations.png",
    )

    def _table(counter: Counter) -> str:
        return "\n".join(f"| {k} | {v} |" for k, v in sorted(counter.items(), key=lambda kv: str(kv[0])))

    discrepancy_section = (
        "No discrepancies — all counts match the official ICBHI totals."
        if not discrepancies
        else "\n".join(
            f"- **{d.dimension}**: expected {d.expected}, got {d.actual}" for d in discrepancies
        )
    )

    duration = inventory_df["duration_s"]
    report = f"""# Data audit report (D05)

Generated from `manifests/dataset_inventory.csv` ({n_recordings} recordings) and
`manifests/cycles.csv` ({n_cycles} cycles).

## Cross-check against official totals
{discrepancy_section}

## Cycle label distribution (4-class)
| label4 | count |
|---|---|
{_table(label_counts)}

## Device distribution (recording-level)
| device | count |
|---|---|
{_table(device_counts)}

## Chest site distribution (recording-level)
| chest_site | count |
|---|---|
{_table(chest_site_counts)}

## Sample rate distribution (recording-level)
| sample_rate | count |
|---|---|
{_table(sample_rate_counts)}

## Duration statistics (seconds, recording-level)
| stat | value |
|---|---|
| min | {duration.min():.2f} |
| max | {duration.max():.2f} |
| mean | {duration.mean():.2f} |
| median | {duration.median():.2f} |

## Figures
![label distribution](figures/labels.png)
![device distribution](figures/devices.png)
![chest site distribution](figures/chest_sites.png)
![duration distribution](figures/durations.png)

## Unique patients
{n_patients} unique patient IDs.
"""
    return report, discrepancies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the D05 data audit report + plots.")
    parser.add_argument("--dataset-inventory", required=True, type=Path)
    parser.add_argument("--cycles", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    inventory_df = pd.read_csv(args.dataset_inventory)
    cycles_df = pd.read_csv(args.cycles)

    report, discrepancies = build_report(inventory_df, cycles_df, args.figures_dir)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report)

    print(f"wrote report: {args.report_out}")
    print(f"wrote 4 figures to: {args.figures_dir}")
    if discrepancies:
        print(f"DISCREPANCIES ({len(discrepancies)}):")
        for d in discrepancies:
            print(f"  {d.dimension}: expected {d.expected}, got {d.actual}")
    else:
        print("no discrepancies vs official totals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

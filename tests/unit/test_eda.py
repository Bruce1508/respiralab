from pathlib import Path

import pandas as pd

from respiralab.data.eda import build_report, compute_discrepancies


def test_no_discrepancy_when_counts_match_official_totals():
    discrepancies = compute_discrepancies(
        n_recordings=920,
        n_patients=126,
        n_cycles=6898,
        label_counts={"normal": 3642, "crackle": 1864, "wheeze": 886, "both": 506},
    )
    assert discrepancies == []


def test_flags_mismatched_patient_count():
    discrepancies = compute_discrepancies(
        n_recordings=920,
        n_patients=125,
        n_cycles=6898,
        label_counts={"normal": 3642, "crackle": 1864, "wheeze": 886, "both": 506},
    )
    assert len(discrepancies) == 1
    assert discrepancies[0].dimension == "patients"
    assert discrepancies[0].expected == 126
    assert discrepancies[0].actual == 125


def test_flags_mismatched_label_count():
    discrepancies = compute_discrepancies(
        n_recordings=920,
        n_patients=126,
        n_cycles=6898,
        label_counts={"normal": 3642, "crackle": 1800, "wheeze": 886, "both": 506},
    )
    dims = {d.dimension for d in discrepancies}
    assert "label:crackle" in dims


def test_build_report_writes_four_figures_and_no_discrepancy_text(tmp_path: Path):
    inventory_df = pd.DataFrame(
        {
            "patient_id": ["101", "101", "102"],
            "device": ["Meditron", "Meditron", "AKGC417L"],
            "chest_site": ["Al", "Pr", "Ar"],
            "sample_rate": [4000, 4000, 44100],
            "duration_s": [20.0, 15.0, 25.0],
        }
    )
    cycles_df = pd.DataFrame({"label4": ["normal", "crackle", "normal"]})
    figures_dir = tmp_path / "figures"
    report, discrepancies = build_report(inventory_df, cycles_df, figures_dir)
    assert (figures_dir / "labels.png").exists()
    assert (figures_dir / "devices.png").exists()
    assert (figures_dir / "chest_sites.png").exists()
    assert (figures_dir / "durations.png").exists()
    assert "## Cycle label distribution" in report
    assert "2 unique patient IDs" in report
    assert len(discrepancies) > 0

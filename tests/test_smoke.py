import subprocess
import sys

import respiralab


def test_import_exposes_version():
    assert respiralab.__version__ == "0.1.0"


def test_cli_version_runs():
    result = subprocess.run(
        [sys.executable, "-m", "respiralab.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "respiralab" in result.stdout

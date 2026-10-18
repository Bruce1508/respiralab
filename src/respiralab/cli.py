"""RespiraLab CLI entry point.

Subcommands are added incrementally as each daily task implements them
(see planning/tasks.json upstream in RespiraLab_Project_Pack). Only `--version`
exists today; audit/split/features/train-ml/train-cnn/evaluate/predict land
in later tasks per docs/02_ARCHITECTURE.md's CLI contract.
"""

import argparse

from respiralab import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="respiralab")
    parser.add_argument("--version", action="version", version=f"respiralab {__version__}")
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

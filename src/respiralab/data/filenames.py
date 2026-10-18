"""Parse the ICBHI 2017 filename convention into structured fields.

Filenames follow: {patient_id}_{recording_index}_{chest_site}_{mode}_{device}.ext
e.g. "101_1b1_Al_sc_Meditron.wav" -> patient 101, recording index "1b1",
chest site Al (anterior left), single-channel ("sc"), Meditron stethoscope.
Pattern and vocabularies (7 chest sites, 2 modes, 4 devices) are confirmed
against both the official page's filename_format.txt and the actual
downloaded archive: all 920 real filenames match with zero exceptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERN = re.compile(
    r"^(?P<patient_id>\d+)_(?P<recording_index>[^_]+)_(?P<chest_site>[A-Za-z]+)_"
    r"(?P<mode>sc|mc)_(?P<device>[A-Za-z0-9]+)$"
)

VALID_CHEST_SITES = frozenset({"Tc", "Al", "Ar", "Pl", "Pr", "Ll", "Lr"})
VALID_DEVICES = frozenset({"AKGC417L", "LittC2SE", "Litt3200", "Meditron"})


@dataclass(frozen=True)
class ParsedFilename:
    record_id: str
    patient_id: str
    recording_index: str
    chest_site: str
    mode: str
    device: str


class FilenameParseError(ValueError):
    """Raised with the offending stem so callers can log rejections by filename."""


def parse_icbhi_filename(stem: str) -> ParsedFilename:
    """Parse a filename stem (no extension) into its ICBHI naming components.

    Raises FilenameParseError if the pattern doesn't match or a field has an
    unexpected value. Callers must catch this and log by filename rather than
    silently skipping the file (docs/03_DATA_AND_EVALUATION_CONTRACT.md).
    """
    m = _PATTERN.match(stem)
    if not m:
        raise FilenameParseError(f"filename does not match ICBHI 5-part pattern: {stem!r}")
    chest_site = m["chest_site"]
    if chest_site not in VALID_CHEST_SITES:
        raise FilenameParseError(f"unknown chest site {chest_site!r} in {stem!r}")
    device = m["device"]
    if device not in VALID_DEVICES:
        raise FilenameParseError(f"unknown device {device!r} in {stem!r}")
    return ParsedFilename(
        record_id=stem,
        patient_id=m["patient_id"],
        recording_index=m["recording_index"],
        chest_site=chest_site,
        mode=m["mode"],
        device=device,
    )

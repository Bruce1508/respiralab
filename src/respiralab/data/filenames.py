from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERN = re.compile(
    "^(?P<patient_id>\\d+)_(?P<recording_index>[^_]+)_(?P<chest_site>[A-Za-z]+)_(?P<mode>sc|mc)_(?P<device>[A-Za-z0-9]+)$"
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


@dataclass
class RejectionRow:
    filename: str
    reason: str


class FilenameParseError(ValueError):
    pass


def parse_icbhi_filename(stem: str) -> ParsedFilename:
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

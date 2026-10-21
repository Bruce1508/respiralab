"""D55: HF_Lung_V1 label-file parser and breath-phase -> 4-class mapping.

Implements exactly the rule fixed in docs/adr/0002-external-label-mapping.md --
no behavior here may diverge from that document; if it needs to change, the ADR
changes first.
"""

from __future__ import annotations

from dataclasses import dataclass

PHASE_TYPES = ("I", "E")
CRACKLE_TYPE = "D"
WHEEZE_TYPE = "Wheeze"
OTHER_ADVENTITIOUS_TYPES = ("Rhonchi", "Stridor")


@dataclass(frozen=True)
class RawEvent:
    type: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class PhaseLabel:
    start_s: float
    end_s: float
    phase_type: str  # "I" or "E"
    label4: str | None  # one of LABEL_ORDER, or None if excluded


def _parse_timestamp(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_label_file(text: str) -> list[RawEvent]:
    """Parses `<type> <start HH:MM:SS.mmm> <end HH:MM:SS.mmm>` lines, skipping blanks."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"malformed label line (expected 3 fields): {line!r}")
        type_, start, end = parts
        events.append(RawEvent(type_, _parse_timestamp(start), _parse_timestamp(end)))
    return events


def _overlaps(a: RawEvent, start_s: float, end_s: float) -> bool:
    return a.start_s < end_s and a.end_s > start_s


def assign_phase_labels(events: list[RawEvent]) -> list[PhaseLabel]:
    """Per docs/adr/0002-external-label-mapping.md: one label per I/E phase, by
    temporal overlap with D/Wheeze/Rhonchi/Stridor events. Rhonchi/Stridor never
    produce a label themselves and never block a D/Wheeze verdict; a phase is
    excluded only when its only overlapping adventitious events are Rhonchi/Stridor.
    """
    phases = [e for e in events if e.type in PHASE_TYPES]
    adventitious = [e for e in events if e.type not in PHASE_TYPES]

    results = []
    for phase in phases:
        has_d = any(a.type == CRACKLE_TYPE and _overlaps(a, phase.start_s, phase.end_s) for a in adventitious)
        has_wheeze = any(a.type == WHEEZE_TYPE and _overlaps(a, phase.start_s, phase.end_s) for a in adventitious)
        has_other = any(
            a.type in OTHER_ADVENTITIOUS_TYPES and _overlaps(a, phase.start_s, phase.end_s) for a in adventitious
        )
        if has_d and has_wheeze:
            label4 = "both"
        elif has_d:
            label4 = "crackle"
        elif has_wheeze:
            label4 = "wheeze"
        elif has_other:
            label4 = None
        else:
            label4 = "normal"
        results.append(PhaseLabel(phase.start_s, phase.end_s, phase.type, label4))
    return results

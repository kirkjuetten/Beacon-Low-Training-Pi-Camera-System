"""Operator/engineer-facing help for ``camera_config.json`` keys.

This module is the single source of truth for "what does this setting do
and where should I start?". It is consumed by:

* :mod:`inspection_system.app.config_validation` -- validation warnings
  cite ``--explain <key>`` so an engineer can read the guidance without
  context-switching to the docs site.
* ``python -m inspection_system --explain <key>`` -- prints one entry.
* ``python -m inspection_system --explain all`` -- prints every entry.

Keep entries short: 4-8 lines each. Anything longer belongs in
``docs/PI_CONNECTION.md`` or a future runbook. Each entry has a
``starting_point`` field with a recommended default that has worked on
the bench; it is *not* a guarantee, just a known-reasonable seed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HelpEntry:
    key: str
    summary: str
    valid: str
    starting_point: str
    symptom: str
    related: tuple[str, ...] = ()


_ENTRIES: tuple[HelpEntry, ...] = (
    HelpEntry(
        key="inspection.threshold_value",
        summary=(
            "Pixel intensity (0-255) that separates 'print' from 'background' when "
            "threshold_mode is 'fixed' or 'fixed_inv'. Ignored for otsu modes."
        ),
        valid="integer 0..255",
        starting_point=(
            "127 for back-lit dark print on a bright background; "
            "raise toward 180-200 for faint print, lower toward 80-100 for very dark print."
        ),
        symptom=(
            "Too high: print disappears (false reject 'missing print'). "
            "Too low: background bleeds in (false reject 'extra print')."
        ),
        related=("inspection.threshold_mode",),
    ),
    HelpEntry(
        key="inspection.threshold_mode",
        summary=(
            "How the binary mask is computed. 'fixed' uses threshold_value as a "
            "hard cutoff. 'otsu' picks the cutoff automatically per image. "
            "The '_inv' variants invert the result for dark-on-light vs light-on-dark."
        ),
        valid="one of: fixed, otsu, fixed_inv, otsu_inv",
        starting_point=(
            "Start with 'otsu' for general bench work; switch to 'fixed' once you have "
            "stable lighting and want repeatability run-to-run."
        ),
        symptom=(
            "If results swing between identical-looking parts, otsu is reacting to glare; "
            "lock to 'fixed' with a measured threshold_value."
        ),
        related=("inspection.threshold_value",),
    ),
    HelpEntry(
        key="inspection.roi.width",
        summary="Width in pixels of the inspection region of interest (ROI).",
        valid="positive integer; must fit inside the captured image",
        starting_point=(
            "Capture once at full resolution, look at the print area, and pick a width "
            "that gives ~10-20% padding around it. Typical: 600-1200 px."
        ),
        symptom=(
            "Zero or negative crashes the pipeline. Too wide pulls in conveyor edges "
            "and noisy lighting; too narrow clips the print and triggers false rejects."
        ),
        related=("inspection.roi.height", "inspection.roi.x", "inspection.roi.y"),
    ),
    HelpEntry(
        key="inspection.roi.height",
        summary="Height in pixels of the inspection ROI.",
        valid="positive integer; must fit inside the captured image",
        starting_point="Same guidance as roi.width; typical 400-800 px.",
        symptom="Same failure modes as roi.width.",
        related=("inspection.roi.width", "inspection.roi.x", "inspection.roi.y"),
    ),
    HelpEntry(
        key="inspection.roi.x",
        summary="Left edge (X origin) of the inspection ROI in pixels.",
        valid="non-negative integer; x + width must be <= image width",
        starting_point=(
            "Capture once and read the X coordinate of the print area's top-left corner; "
            "subtract ~10% of width for padding."
        ),
        symptom=(
            "Off-by-a-lot causes 'registration failure' or alignment to drift onto "
            "background; off-by-a-little just shifts scoring slightly."
        ),
        related=("inspection.roi.width",),
    ),
    HelpEntry(
        key="inspection.roi.y",
        summary="Top edge (Y origin) of the inspection ROI in pixels.",
        valid="non-negative integer; y + height must be <= image height",
        starting_point="Same guidance as roi.x.",
        symptom="Same failure modes as roi.x.",
        related=("inspection.roi.height",),
    ),
    HelpEntry(
        key="io.mode",
        summary=(
            "Indicator stack driver. 'none' disables hardware indicators (safe for "
            "bench/dev). 'gpio' drives Pi GPIO pins directly. 'modbus' speaks Modbus "
            "RTU over the Waveshare USB-RS485 converter."
        ),
        valid="one of: none, gpio, modbus",
        starting_point=(
            "'none' until the RS-485 dongle and IO module are wired and verified with "
            "--self-test. Then 'modbus' for production."
        ),
        symptom=(
            "Wrong mode prints 'Indicator I/O error' in the end-of-session summary; "
            "the inspection itself still records correctly."
        ),
        related=("io.modbus.slave_id", "io.modbus.parity"),
    ),
    HelpEntry(
        key="io.modbus.slave_id",
        summary="Modbus RTU slave address of the IO module on the bus.",
        valid="integer 1..247",
        starting_point=(
            "1 for the 8CH digital-input module out of the box. 2 if you chain a "
            "4CH relay board on the same A/B pair."
        ),
        symptom=(
            "Wrong ID looks like silent failure: every pulse times out. Confirm with "
            "the module's DIP switches and a quick read on the Pi."
        ),
        related=("io.mode", "io.modbus.parity"),
    ),
    HelpEntry(
        key="io.modbus.parity",
        summary="Serial parity for the RS-485 link.",
        valid="one of: N, E, O",
        starting_point="N (matches the Waveshare module factory default at 9600 8N1).",
        symptom=(
            "Mismatched parity causes intermittent reads and a non-zero "
            "'Indicator I/O errors' count in the session summary."
        ),
        related=("io.mode", "io.modbus.slave_id"),
    ),
    HelpEntry(
        key="pilot_readiness.enforce",
        summary=(
            "When true (default), production mode refuses to launch on a project "
            "that has not finished commissioning. When false, the dashboard launches "
            "anyway and prints a warning."
        ),
        valid="boolean",
        starting_point="true. Only flip to false for engineering bring-up runs.",
        symptom=(
            "If production won't launch and the operator sees a readiness report, the "
            "fix is to finish the missing commissioning step, not to disable the gate."
        ),
    ),
)


_BY_KEY: dict[str, HelpEntry] = {e.key: e for e in _ENTRIES}


def all_entries() -> tuple[HelpEntry, ...]:
    return _ENTRIES


def get_entry(key: str) -> HelpEntry | None:
    """Return the entry for ``key`` or ``None`` if unknown.

    Lookup is case-sensitive on the dotted path because that is the form
    that appears in validation messages and config files.
    """
    return _BY_KEY.get(key)


def format_entry(entry: HelpEntry) -> str:
    lines = [
        entry.key,
        "-" * len(entry.key),
        f"What it does : {entry.summary}",
        f"Valid values : {entry.valid}",
        f"Start with   : {entry.starting_point}",
        f"If wrong     : {entry.symptom}",
    ]
    if entry.related:
        lines.append(f"Related      : {', '.join(entry.related)}")
    return "\n".join(lines)


def format_all() -> str:
    return "\n\n".join(format_entry(e) for e in _ENTRIES)


def hint_for(key: str) -> str:
    """Short suffix appended to validation messages.

    Returns a non-empty string only when we have a help entry for the key,
    so callers can safely concatenate.
    """
    if key in _BY_KEY:
        return f" (run: python -m inspection_system --explain {key})"
    return ""

"""Lightweight, dependency-free validation for ``camera_config.json``.

Why not jsonschema?
-------------------
The Pi runs the inspection system with a minimal Python install (opencv,
numpy, pygame). Adding ``jsonschema`` to the runtime path means another
dependency to keep current on a fleet of devices for a check that, in
practice, only needs to flag a small number of well-known mistakes:

* ``inspection.threshold_value`` outside ``0..255`` (clips silently otherwise).
* ``inspection.roi.width`` / ``height`` non-positive (causes opaque downstream
  ``cv2.cvtColor`` asserts -- the same class of bug that motivated the
  Phase 0 hotfix to ``get_roi``).
* ``io.mode`` set to a value the dispatcher does not recognize.
* ``io.modbus.parity`` set to anything other than ``"N" | "E" | "O"``.

This module returns a list of human-readable issues rather than raising,
so :func:`load_config` can print them without breaking already-deployed
recipes that have a benign typo. Promote to a hard error in a later phase
once the field is clean.
"""
from __future__ import annotations

from typing import Any, Iterable


_VALID_IO_MODES = {"none", "gpio", "modbus"}
_VALID_MODBUS_PARITY = {"N", "E", "O"}
_VALID_THRESHOLD_MODES = {"fixed", "otsu", "fixed_inv", "otsu_inv"}


def _get(d: Any, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def _check_int_range(
    issues: list[str], value: Any, label: str, lo: int, hi: int
) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(f"{label}: expected integer, got {type(value).__name__}")
        return
    if value < lo or value > hi:
        issues.append(f"{label}: {value} is outside {lo}..{hi}")


def _check_positive_int(issues: list[str], value: Any, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        issues.append(f"{label}: expected positive integer, got {value!r}")


def validate_config(config: Any) -> list[str]:
    """Return a list of issues found in ``config``. Empty list = clean."""
    issues: list[str] = []
    if not isinstance(config, dict):
        return ["config: top-level value must be an object"]

    inspection = _get(config, "inspection", {})
    if not isinstance(inspection, dict):
        issues.append("inspection: must be an object")
        inspection = {}

    _check_int_range(
        issues, inspection.get("threshold_value"), "inspection.threshold_value", 0, 255
    )

    threshold_mode = inspection.get("threshold_mode")
    if threshold_mode is not None and threshold_mode not in _VALID_THRESHOLD_MODES:
        issues.append(
            f"inspection.threshold_mode: {threshold_mode!r} not in {sorted(_VALID_THRESHOLD_MODES)}"
        )

    roi = inspection.get("roi", {})
    if not isinstance(roi, dict):
        issues.append("inspection.roi: must be an object")
    else:
        for axis in ("width", "height"):
            _check_positive_int(issues, roi.get(axis), f"inspection.roi.{axis}")
        for axis in ("x", "y"):
            value = roi.get(axis)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                issues.append(f"inspection.roi.{axis}: expected non-negative integer, got {value!r}")

    io_block = _get(config, "io", {})
    if not isinstance(io_block, dict):
        issues.append("io: must be an object")
    else:
        mode = io_block.get("mode")
        if mode is not None and str(mode).lower() not in _VALID_IO_MODES:
            issues.append(f"io.mode: {mode!r} not in {sorted(_VALID_IO_MODES)}")

        modbus = io_block.get("modbus", {})
        if isinstance(modbus, dict):
            parity = modbus.get("parity")
            if parity is not None and parity not in _VALID_MODBUS_PARITY:
                issues.append(
                    f"io.modbus.parity: {parity!r} not in {sorted(_VALID_MODBUS_PARITY)}"
                )
            _check_int_range(issues, modbus.get("slave_id"), "io.modbus.slave_id", 1, 247)
            for ch in ("pass_channel", "fail_channel"):
                value = modbus.get(ch)
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                    issues.append(f"io.modbus.{ch}: expected non-negative integer, got {value!r}")

    return issues


def format_issues(issues: Iterable[str]) -> str:
    """Render issues as a single multi-line string for log output."""
    return "\n".join(f"  - {issue}" for issue in issues)

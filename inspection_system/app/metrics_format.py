"""Shared helpers for formatting inspection metrics into log/UI text.

Background
----------
Several modules (the training logger, the diagnostics writer, the result
interpreter) all need to render numeric metrics that may legitimately be
``None`` for a given inspection -- for example, ``ssim`` is only present
when SSIM scoring is enabled. The historical pattern was an inline check
in each call site, which made it easy to forget the guard and get a
``TypeError: unsupported format string passed to NoneType.__format__`` at
runtime in operator-facing log lines.

This module centralizes the guard so callers can write::

    log_entry = append_optional_metric(
        log_entry, details, "ssim", " | SSIM: {value:.3f}"
    )

and ``None`` simply leaves the log entry unchanged.
"""
from __future__ import annotations

from typing import Mapping


def append_optional_metric(
    log_entry: str, details: Mapping[str, object], key: str, template: str
) -> str:
    """Append ``template.format(value=details[key])`` if the value is not None.

    The template should use the named field ``{value}`` and any standard
    format spec, e.g. ``" | SSIM: {value:.3f}"``. If ``details[key]`` is
    missing or ``None``, ``log_entry`` is returned unchanged.
    """
    value = details.get(key) if isinstance(details, Mapping) else None
    if value is None:
        return log_entry
    return log_entry + template.format(value=value)


def safe_format_float(value: object, spec: str = ".2f", fallback: str = "n/a") -> str:
    """Format ``value`` as a float using ``spec``; return ``fallback`` on failure.

    Use at presentation boundaries (operator log lines, UI labels) where a
    crash would be worse than a graceful ``"n/a"`` placeholder. For internal
    code paths that should never receive ``None``, prefer an explicit check.
    """
    if value is None:
        return fallback
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return fallback

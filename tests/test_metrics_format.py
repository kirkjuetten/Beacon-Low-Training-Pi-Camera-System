"""Tests for inspection_system.app.metrics_format."""
from __future__ import annotations

import pytest

from inspection_system.app.metrics_format import (
    append_optional_metric,
    safe_format_float,
)


# --- append_optional_metric -------------------------------------------------


def test_append_optional_metric_appends_when_value_present():
    out = append_optional_metric("base", {"ssim": 0.987}, "ssim", " | SSIM: {value:.3f}")
    assert out == "base | SSIM: 0.987"


def test_append_optional_metric_returns_unchanged_when_value_is_none():
    out = append_optional_metric("base", {"ssim": None}, "ssim", " | SSIM: {value:.3f}")
    assert out == "base"


def test_append_optional_metric_returns_unchanged_when_key_missing():
    out = append_optional_metric("base", {}, "ssim", " | SSIM: {value:.3f}")
    assert out == "base"


def test_append_optional_metric_returns_unchanged_when_details_not_mapping():
    # The historical inline guard would crash here; the helper must not.
    out = append_optional_metric("base", None, "ssim", " | SSIM: {value:.3f}")  # type: ignore[arg-type]
    assert out == "base"


def test_append_optional_metric_supports_percent_format():
    out = append_optional_metric("base", {"k": 0.25}, "k", " | k: {value:.1%}")
    assert out == "base | k: 25.0%"


# --- safe_format_float ------------------------------------------------------


def test_safe_format_float_formats_numeric_values():
    assert safe_format_float(1.234, ".2f") == "1.23"
    assert safe_format_float(7, ".1f") == "7.0"
    assert safe_format_float("3.14", ".2f") == "3.14"


def test_safe_format_float_returns_fallback_for_none():
    assert safe_format_float(None) == "n/a"
    assert safe_format_float(None, fallback="-") == "-"


def test_safe_format_float_returns_fallback_for_unparseable():
    assert safe_format_float("nope") == "n/a"
    assert safe_format_float(object()) == "n/a"


# --- TrainingLogger re-export back-compat -----------------------------------


def test_training_logger_reexported_from_interactive_training():
    from inspection_system.app import interactive_training
    from inspection_system.app.training.logger import TrainingLogger as Canonical

    assert interactive_training.TrainingLogger is Canonical


def test_training_logger_static_method_still_callable(tmp_path):
    """The legacy ``TrainingLogger._append_optional_metric`` staticmethod
    still works for any external code that bound to it before Phase 4."""
    from inspection_system.app.training.logger import TrainingLogger

    out = TrainingLogger._append_optional_metric(
        "base", {"ssim": 0.5}, "ssim", " | SSIM: {value:.2f}"
    )
    assert out == "base | SSIM: 0.50"

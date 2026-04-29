"""Tests for inspection_system.app.config_validation."""
from __future__ import annotations

from inspection_system.app.config_validation import (
    format_issues,
    validate_config,
)


def test_validate_config_clean_config_returns_empty():
    config = {
        "inspection": {
            "threshold_value": 180,
            "threshold_mode": "otsu",
            "roi": {"x": 0, "y": 0, "width": 100, "height": 100},
        },
        "io": {
            "mode": "modbus",
            "modbus": {"parity": "N", "slave_id": 1, "pass_channel": 0, "fail_channel": 1},
        },
    }
    assert validate_config(config) == []


def test_validate_config_rejects_top_level_non_object():
    assert validate_config([1, 2, 3]) == ["config: top-level value must be an object"]


def test_validate_config_flags_threshold_value_out_of_range():
    issues = validate_config({"inspection": {"threshold_value": 999}})
    assert any("threshold_value" in i and "0..255" in i for i in issues)


def test_validate_config_flags_threshold_value_wrong_type():
    issues = validate_config({"inspection": {"threshold_value": "high"}})
    assert any("threshold_value" in i and "expected integer" in i for i in issues)


def test_validate_config_flags_unknown_threshold_mode():
    issues = validate_config({"inspection": {"threshold_mode": "magic"}})
    assert any("threshold_mode" in i for i in issues)


def test_validate_config_flags_non_positive_roi_dimensions():
    issues = validate_config({"inspection": {"roi": {"width": 0, "height": -5}}})
    assert any("roi.width" in i for i in issues)
    assert any("roi.height" in i for i in issues)


def test_validate_config_flags_negative_roi_origin():
    issues = validate_config({"inspection": {"roi": {"x": -1, "y": 0, "width": 10, "height": 10}}})
    assert any("roi.x" in i for i in issues)


def test_validate_config_flags_unknown_io_mode():
    issues = validate_config({"io": {"mode": "bluetooth"}})
    assert any("io.mode" in i for i in issues)


def test_validate_config_flags_invalid_modbus_parity():
    issues = validate_config({"io": {"modbus": {"parity": "Z"}}})
    assert any("modbus.parity" in i for i in issues)


def test_validate_config_flags_modbus_slave_id_out_of_range():
    issues = validate_config({"io": {"modbus": {"slave_id": 999}}})
    assert any("slave_id" in i and "1..247" in i for i in issues)


def test_validate_config_flags_negative_modbus_channel():
    issues = validate_config({"io": {"modbus": {"pass_channel": -1}}})
    assert any("pass_channel" in i for i in issues)


def test_validate_config_tolerates_missing_sections():
    # An empty config (no inspection / io) is the legitimate first-run case.
    assert validate_config({}) == []


def test_format_issues_renders_bullet_list():
    rendered = format_issues(["a", "b"])
    assert rendered == "  - a\n  - b"


# --- segmentation back-compat shims -----------------------------------------


def test_segmentation_shims_reexport_canonical_callables():
    from inspection_system.app.segmentation import binary_threshold, binary_threshold_inverted
    from inspection_system.app.segmentation.thresholding import (
        apply_binary_threshold,
        apply_binary_threshold_inverted,
    )

    assert binary_threshold.apply_binary_threshold is apply_binary_threshold
    assert (
        binary_threshold_inverted.apply_binary_threshold_inverted
        is apply_binary_threshold_inverted
    )

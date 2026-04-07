import time

import pytest

from inspection_system.app.operator_dashboard import (
    apply_config_updates,
    build_config_editor_values,
    find_preview_image,
    get_nested_config_value,
    parse_config_value,
)


def test_parse_config_value_supports_bool_int_and_float() -> None:
    assert parse_config_value("true", bool) is True
    assert parse_config_value("0", bool) is False
    assert parse_config_value("42", int) == 42
    assert parse_config_value("0.125", float) == 0.125


def test_parse_config_value_rejects_invalid_bool() -> None:
    with pytest.raises(ValueError):
        parse_config_value("maybe", bool)


def test_apply_config_updates_updates_nested_fields() -> None:
    config = {
        "capture": {"timeout_ms": 200},
        "inspection": {
            "threshold_value": 180,
            "min_required_coverage": 0.92,
            "save_debug_images": True,
        },
        "alignment": {"enabled": True},
        "indicator_led": {"enabled": False},
    }

    updated = apply_config_updates(
        config,
        {
            "capture.timeout_ms": "350",
            "inspection.min_required_coverage": "0.975",
            "inspection.save_debug_images": "false",
            "alignment.enabled": "false",
        },
    )

    assert get_nested_config_value(updated, "capture.timeout_ms") == 350
    assert get_nested_config_value(updated, "inspection.min_required_coverage") == 0.975
    assert get_nested_config_value(updated, "inspection.save_debug_images") is False
    assert get_nested_config_value(updated, "alignment.enabled") is False
    assert get_nested_config_value(config, "capture.timeout_ms") == 200


def test_build_config_editor_values_returns_string_values() -> None:
    config = {
        "capture": {"timeout_ms": 200},
        "inspection": {"min_required_coverage": 0.92},
        "alignment": {"enabled": True},
        "indicator_led": {"enabled": False},
    }

    values = build_config_editor_values(config)

    assert values["capture.timeout_ms"] == "200"
    assert values["inspection.min_required_coverage"] == "0.92"
    assert values["alignment.enabled"] == "True"
    assert values["indicator_led.enabled"] == "False"


def test_find_preview_image_returns_most_recent_image(tmp_path) -> None:
    older = tmp_path / "golden_reference_image.png"
    newer = tmp_path / "sample_diff.png"
    ignored = tmp_path / "notes.txt"

    older.write_bytes(b"older")
    ignored.write_text("ignore", encoding="utf-8")
    newer.write_bytes(b"newer")
    now = time.time()
    older_time = now - 10
    newer_time = now - 1
    older.touch()
    newer.touch()
    older.chmod(0o666)
    newer.chmod(0o666)
    import os
    os.utime(older, (older_time, older_time))
    os.utime(newer, (newer_time, newer_time))

    preview = find_preview_image(tmp_path)

    assert preview == newer


def test_find_preview_image_returns_none_without_supported_files(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert find_preview_image(tmp_path) is None
import time
from pathlib import Path

import pytest

from inspection_system.app.operator_dashboard import (
    apply_config_updates,
    build_config_editor_values,
    describe_preview_image,
    find_preview_image,
    get_nested_config_value,
    parse_config_value,
    should_close_dashboard_on_launch,
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
    older = tmp_path / "capture_a.png"
    newer = tmp_path / "capture_b.png"
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


def test_find_preview_image_prefers_live_capture_preview(tmp_path) -> None:
    live = tmp_path / "dashboard_live_preview.png"
    reference = tmp_path / "golden_reference_image.png"

    live.write_bytes(b"live")
    reference.write_bytes(b"reference")

    preview = find_preview_image(tmp_path, is_informative_fn=lambda _p: True)

    assert preview == live


def test_find_preview_image_prefers_reference_image_over_newer_debug_diff(tmp_path) -> None:
    reference = tmp_path / "golden_reference_image.png"
    debug_diff = tmp_path / "temp_capture_diff.png"

    reference.write_bytes(b"reference")
    debug_diff.write_bytes(b"diff")

    now = time.time()
    old_time = now - 10
    new_time = now - 1
    import os

    os.utime(reference, (old_time, old_time))
    os.utime(debug_diff, (new_time, new_time))

    preview = find_preview_image(tmp_path)

    assert preview == reference


def test_find_preview_image_skips_non_informative_reference(tmp_path) -> None:
    reference = tmp_path / "golden_reference_image.png"
    sample = tmp_path / "capture_latest.png"

    reference.write_bytes(b"reference")
    sample.write_bytes(b"sample")

    now = time.time()
    old_time = now - 10
    new_time = now - 1
    import os

    os.utime(reference, (old_time, old_time))
    os.utime(sample, (new_time, new_time))

    informative = {
        reference.name: False,
        sample.name: True,
    }
    preview = find_preview_image(tmp_path, is_informative_fn=lambda p: informative.get(p.name, True))

    assert preview == sample


def test_find_preview_image_uses_informative_debug_if_needed(tmp_path) -> None:
    sample = tmp_path / "capture_latest.png"
    debug_diff = tmp_path / "temp_capture_diff.png"

    sample.write_bytes(b"sample")
    debug_diff.write_bytes(b"diff")

    now = time.time()
    old_time = now - 10
    new_time = now - 1
    import os

    os.utime(sample, (old_time, old_time))
    os.utime(debug_diff, (new_time, new_time))

    informative = {
        sample.name: False,
        debug_diff.name: True,
    }
    preview = find_preview_image(tmp_path, is_informative_fn=lambda p: informative.get(p.name, True))

    assert preview == debug_diff


def test_find_preview_image_returns_none_without_supported_files(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert find_preview_image(tmp_path) is None


def test_should_close_dashboard_on_launch_policy() -> None:
    assert should_close_dashboard_on_launch("project-manager") is True
    assert should_close_dashboard_on_launch("config-editor") is True
    assert should_close_dashboard_on_launch("train") is False
    assert should_close_dashboard_on_launch("capture") is False


def test_describe_preview_image_categories() -> None:
    assert describe_preview_image(Path("dashboard_live_preview.png")) == "live capture"
    assert describe_preview_image(Path("golden_reference_image.png")) == "reference"
    assert describe_preview_image(Path("temp_capture_diff.png")) == "difference debug"
    assert describe_preview_image(Path("temp_capture_mask.png")) == "mask debug"
    assert describe_preview_image(Path("capture_latest.png")) == "latest sample"
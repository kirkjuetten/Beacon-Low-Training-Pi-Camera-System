import json
import time
from pathlib import Path

import pytest

from inspection_system.app.config_service import (
    CONFIG_DROPDOWN_OPTIONS,
    CONFIG_FIELD_SPECS,
    apply_config_updates,
    build_config_editor_values,
    get_nested_config_value,
    parse_config_value,
)
from inspection_system.app.operator_dashboard import (
    build_dashboard_hint_text,
    should_use_compact_layout,
    should_close_dashboard_on_launch,
)
from inspection_system.app.preview_service import describe_preview_image, find_preview_image


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
            "inspection_mode": "mask_only",
            "threshold_mode": "otsu",
            "threshold_value": 180,
            "min_required_coverage": 0.92,
            "max_mean_edge_distance_px": None,
            "max_section_edge_distance_px": None,
            "max_section_width_delta_ratio": None,
            "max_section_center_offset_px": None,
            "save_debug_images": True,
        },
        "alignment": {
            "enabled": True,
            "mode": "moments",
            "registration": {
                "strategy": "moments",
                "transform_model": "rigid",
                "anchor_mode": "none",
                "subpixel_refinement": "off",
                "search_margin_px": 24,
                "quality_gates": {
                    "min_confidence": None,
                    "max_mean_residual_px": None,
                },
                "datum_frame": {
                    "origin": "roi_top_left",
                    "orientation": "part_axis",
                },
            },
        },
        "indicator_led": {"enabled": False},
    }

    updated = apply_config_updates(
        config,
        {
            "capture.timeout_ms": "350",
            "inspection.inspection_mode": "full",
            "inspection.reference_strategy": "hybrid",
            "inspection.blend_mode": "blend_balanced",
            "inspection.tolerance_mode": "forgiving",
            "inspection.threshold_mode": "fixed_inv",
            "inspection.min_required_coverage": "0.975",
            "inspection.max_mean_edge_distance_px": "1.25",
            "inspection.max_section_edge_distance_px": "0.75",
            "inspection.max_section_width_delta_ratio": "0.12",
            "inspection.max_section_center_offset_px": "0.6",
            "inspection.save_debug_images": "false",
            "alignment.enabled": "false",
            "alignment.mode": "anchor_pair",
            "alignment.registration.strategy": "anchor_pair",
            "alignment.registration.transform_model": "similarity",
            "alignment.registration.anchor_mode": "pair",
            "alignment.registration.subpixel_refinement": "phase_correlation",
            "alignment.registration.search_margin_px": "32",
            "alignment.registration.quality_gates.min_confidence": "0.91",
            "alignment.registration.quality_gates.max_mean_residual_px": "1.4",
            "alignment.registration.datum_frame.origin": "anchor_primary",
            "alignment.registration.datum_frame.orientation": "anchor_pair",
        },
    )

    assert get_nested_config_value(updated, "capture.timeout_ms") == 350
    assert get_nested_config_value(updated, "inspection.inspection_mode") == "full"
    assert get_nested_config_value(updated, "inspection.reference_strategy") == "hybrid"
    assert get_nested_config_value(updated, "inspection.blend_mode") == "blend_balanced"
    assert get_nested_config_value(updated, "inspection.tolerance_mode") == "forgiving"
    assert get_nested_config_value(updated, "inspection.threshold_mode") == "fixed_inv"
    assert get_nested_config_value(updated, "inspection.min_required_coverage") == 0.975
    assert get_nested_config_value(updated, "inspection.max_mean_edge_distance_px") == 1.25
    assert get_nested_config_value(updated, "inspection.max_section_edge_distance_px") == 0.75
    assert get_nested_config_value(updated, "inspection.max_section_width_delta_ratio") == 0.12
    assert get_nested_config_value(updated, "inspection.max_section_center_offset_px") == 0.6
    assert get_nested_config_value(updated, "inspection.save_debug_images") is False
    assert get_nested_config_value(updated, "alignment.enabled") is False
    assert get_nested_config_value(updated, "alignment.mode") == "anchor_pair"
    assert get_nested_config_value(updated, "alignment.registration.strategy") == "anchor_pair"
    assert get_nested_config_value(updated, "alignment.registration.transform_model") == "similarity"
    assert get_nested_config_value(updated, "alignment.registration.anchor_mode") == "pair"
    assert get_nested_config_value(updated, "alignment.registration.subpixel_refinement") == "phase_correlation"
    assert get_nested_config_value(updated, "alignment.registration.search_margin_px") == 32
    assert get_nested_config_value(updated, "alignment.registration.quality_gates.min_confidence") == 0.91
    assert get_nested_config_value(updated, "alignment.registration.quality_gates.max_mean_residual_px") == 1.4
    assert get_nested_config_value(updated, "alignment.registration.datum_frame.origin") == "anchor_primary"
    assert get_nested_config_value(updated, "alignment.registration.datum_frame.orientation") == "anchor_pair"
    assert get_nested_config_value(config, "capture.timeout_ms") == 200


def test_apply_config_updates_ignores_blank_non_optional_values() -> None:
    config = {
        "capture": {},
        "inspection": {"threshold_mode": "otsu"},
        "alignment": {},
        "indicator_led": {},
    }

    updated = apply_config_updates(
        config,
        {
            "capture.timeout_ms": "",
            "capture.shutter_us": "",
            "inspection.threshold_mode": "fixed",
            "inspection.threshold_value": "",
            "inspection.save_debug_images": "",
            "alignment.registration.quality_gates.min_confidence": "",
        },
    )

    assert get_nested_config_value(updated, "capture.timeout_ms") is None
    assert get_nested_config_value(updated, "capture.shutter_us") is None
    assert get_nested_config_value(updated, "inspection.threshold_mode") == "fixed"
    assert get_nested_config_value(updated, "inspection.threshold_value") is None
    assert get_nested_config_value(updated, "inspection.save_debug_images") is None
    assert get_nested_config_value(updated, "alignment.registration.quality_gates.min_confidence") is None


def test_build_config_editor_values_returns_string_values() -> None:
    config = {
        "capture": {"timeout_ms": 200},
        "inspection": {
            "inspection_mode": "mask_only",
            "reference_strategy": "golden_only",
            "blend_mode": "hard_only",
            "tolerance_mode": "balanced",
            "threshold_mode": "otsu",
            "min_required_coverage": 0.92,
            "max_mean_edge_distance_px": 1.5,
            "max_section_edge_distance_px": 0.8,
            "max_section_width_delta_ratio": 0.1,
            "max_section_center_offset_px": 0.5,
        },
        "alignment": {"enabled": True},
        "indicator_led": {"enabled": False},
    }

    values = build_config_editor_values(config)

    assert values["capture.timeout_ms"] == "200"
    assert values["inspection.inspection_mode"] == "mask_only"
    assert values["inspection.reference_strategy"] == "golden_only"
    assert values["inspection.blend_mode"] == "hard_only"
    assert values["inspection.tolerance_mode"] == "balanced"
    assert values["inspection.threshold_mode"] == "otsu"
    assert values["inspection.min_required_coverage"] == "0.92"
    assert values["inspection.max_mean_edge_distance_px"] == "1.5"
    assert values["inspection.max_section_edge_distance_px"] == "0.8"
    assert values["inspection.max_section_width_delta_ratio"] == "0.1"
    assert values["inspection.max_section_center_offset_px"] == "0.5"
    assert values["alignment.enabled"] == "True"
    assert values["alignment.mode"] == ""
    assert values["alignment.registration.strategy"] == ""
    assert values["alignment.registration.quality_gates.min_confidence"] == ""
    assert values["indicator_led.enabled"] == "False"


def test_build_config_editor_values_includes_registration_scalar_fields() -> None:
    config = {
        "alignment": {
            "enabled": True,
            "mode": "anchor_pair",
            "registration": {
                "strategy": "anchor_pair",
                "transform_model": "similarity",
                "anchor_mode": "pair",
                "subpixel_refinement": "phase_correlation",
                "search_margin_px": 28,
                "quality_gates": {
                    "min_confidence": 0.87,
                    "max_mean_residual_px": 1.2,
                },
                "datum_frame": {
                    "origin": "anchor_primary",
                    "orientation": "anchor_pair",
                },
            },
        }
    }

    values = build_config_editor_values(config)

    assert values["alignment.mode"] == "anchor_pair"
    assert values["alignment.registration.strategy"] == "anchor_pair"
    assert values["alignment.registration.transform_model"] == "similarity"
    assert values["alignment.registration.anchor_mode"] == "pair"
    assert values["alignment.registration.subpixel_refinement"] == "phase_correlation"
    assert values["alignment.registration.search_margin_px"] == "28"
    assert values["alignment.registration.quality_gates.min_confidence"] == "0.87"
    assert values["alignment.registration.quality_gates.max_mean_residual_px"] == "1.2"
    assert values["alignment.registration.datum_frame.origin"] == "anchor_primary"
    assert values["alignment.registration.datum_frame.orientation"] == "anchor_pair"


def test_build_dashboard_hint_text_reports_inactive_edge_gate() -> None:
    hint = build_dashboard_hint_text(
        {
            "inspection": {
                "max_mean_edge_distance_px": 1.2,
                "max_section_edge_distance_px": None,
                "max_section_width_delta_ratio": None,
                "max_section_center_offset_px": None,
            }
        }
    )

    assert "Edge Gates: global<=1.20px | section off" in hint
    assert "Width Gate: section off" in hint
    assert "Center Gate: section off" in hint
    assert "set Max Section Edge Distance" in hint
    assert "set Max Section Width Drift" in hint
    assert "set Max Section Center Offset" in hint


def test_build_dashboard_hint_text_includes_commissioning_status_when_runtime_paths_are_available(tmp_path) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "camera_config.json"
    config_file.write_text(json.dumps({"inspection": {}}, indent=2) + "\n", encoding="utf-8")
    (reference_dir / "reference_mask.png").write_bytes(b"mask")
    (reference_dir / "golden_reference_image.png").write_bytes(b"image")
    (tmp_path / "training_data.json").write_text(
        json.dumps([
            {"feedback": "approve", "final_class": "good", "learning_state": "committed"},
            {"feedback": "approve", "final_class": "good", "learning_state": "pending"},
        ], indent=2) + "\n",
        encoding="utf-8",
    )

    hint = build_dashboard_hint_text(
        {
            "inspection": {
                "reference_strategy": "golden_only",
                "max_mean_edge_distance_px": None,
                "max_section_edge_distance_px": None,
                "max_section_width_delta_ratio": None,
                "max_section_center_offset_px": None,
            }
        },
        {
            "config_file": config_file,
            "reference_dir": reference_dir,
            "reference_mask": reference_dir / "reference_mask.png",
            "reference_image": reference_dir / "golden_reference_image.png",
        },
    )

    assert "Workflow: Stage 2/5 - Capture Registration Baseline" in hint
    assert "Instruction: Press Update to commit 1 pending approved-good parts." in hint
    assert "Commissioning: NOT READY | golden ok | reg ok | baseline pending | good 1/10" in hint
    assert "Next: Press Update to commit 1 pending approved-good parts." in hint
    assert "Edge Gates: global off | section off" in hint


def test_dropdown_options_cover_expected_fixed_choice_fields() -> None:
    field_paths = {field for field, _, _ in CONFIG_FIELD_SPECS}

    assert "inspection.inspection_mode" in CONFIG_DROPDOWN_OPTIONS
    assert "inspection.reference_strategy" in CONFIG_DROPDOWN_OPTIONS
    assert "inspection.blend_mode" in CONFIG_DROPDOWN_OPTIONS
    assert "inspection.tolerance_mode" in CONFIG_DROPDOWN_OPTIONS
    assert "inspection.threshold_mode" in CONFIG_DROPDOWN_OPTIONS
    assert "inspection.save_debug_images" in CONFIG_DROPDOWN_OPTIONS
    assert "alignment.enabled" in CONFIG_DROPDOWN_OPTIONS
    assert "alignment.mode" in CONFIG_DROPDOWN_OPTIONS
    assert "alignment.registration.strategy" in CONFIG_DROPDOWN_OPTIONS
    assert "alignment.registration.transform_model" in CONFIG_DROPDOWN_OPTIONS
    assert "alignment.registration.anchor_mode" in CONFIG_DROPDOWN_OPTIONS
    assert "alignment.registration.subpixel_refinement" in CONFIG_DROPDOWN_OPTIONS
    assert "alignment.registration.datum_frame.origin" in CONFIG_DROPDOWN_OPTIONS
    assert "alignment.registration.datum_frame.orientation" in CONFIG_DROPDOWN_OPTIONS
    assert "indicator_led.enabled" in CONFIG_DROPDOWN_OPTIONS
    assert set(CONFIG_DROPDOWN_OPTIONS).issubset(field_paths)


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


def test_find_preview_image_prefers_reference_image(tmp_path) -> None:
    # Live preview is now transient (not persisted); reference image is preferred
    reference = tmp_path / "golden_reference_image.png"
    sample = tmp_path / "capture_latest.png"

    reference.write_bytes(b"reference")
    sample.write_bytes(b"sample")

    preview = find_preview_image(tmp_path, is_informative_fn=lambda _p: True)

    assert preview == reference


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
    assert should_close_dashboard_on_launch("config-editor") is False
    assert should_close_dashboard_on_launch("train") is False
    assert should_close_dashboard_on_launch("capture") is False


def test_should_use_compact_layout_for_small_pi_screens() -> None:
    assert should_use_compact_layout(800, 480) is True
    assert should_use_compact_layout(1024, 600) is True
    assert should_use_compact_layout(1280, 720) is False


def test_describe_preview_image_categories() -> None:
    # Live preview is now transient (not persisted to disk)
    assert describe_preview_image(Path("golden_reference_image.png")) == "reference"
    assert describe_preview_image(Path("temp_capture_diff.png")) == "difference debug"
    assert describe_preview_image(Path("temp_capture_mask.png")) == "mask debug"
    assert describe_preview_image(Path("capture_latest.png")) == "latest sample"


def test_config_editor_depends_on_shared_services_not_dashboard() -> None:
    source_path = Path(__file__).resolve().parents[1] / "inspection_system" / "app" / "config_editor_page.py"
    source_text = source_path.read_text(encoding="utf-8")

    assert "from inspection_system.app.operator_dashboard import" not in source_text
    assert "from inspection_system.app.config_service import" in source_text
    assert "from inspection_system.app.preview_service import" in source_text
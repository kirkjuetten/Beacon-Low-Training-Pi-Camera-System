import json
from pathlib import Path
from unittest import mock

from inspection_system.app.runtime_controller import (
    describe_edge_gate_status,
    describe_section_center_gate_status,
    describe_section_width_gate_status,
    format_operator_mode_lines,
    get_commissioning_status,
    get_inspection_runtime_warnings,
    print_inspection_result,
)


def _default_registration_baseline() -> dict:
    return {
        "runtime_mode": "moments",
        "strategy": "moments",
        "transform_model": "rigid",
        "anchor_mode": "none",
        "required_anchor_count": 0,
        "enabled_anchor_count": 0,
        "anchor_ids": [],
        "search_margin_px": 24,
        "datum_origin": "roi_top_left",
        "datum_orientation": "part_axis",
        "requires_datum_confirmation": False,
        "datum_confirmed": False,
        "requires_expected_transform_validation": False,
        "expected_transform_confirmed": False,
        "expected_transform": {"max_angle_deg": 1.0, "max_shift_x": 4, "max_shift_y": 3},
        "quality_gates": {"min_confidence": None, "max_mean_residual_px": None},
        "checklist": [
            {"key": "anchors", "label": "Anchor placement", "required": False, "ready": True, "summary": "not required"},
            {"key": "search_windows", "label": "Search windows", "required": False, "ready": True, "summary": "not required"},
            {"key": "datum", "label": "Datum confirmation", "required": False, "ready": True, "summary": "default datum"},
            {"key": "expected_transform", "label": "Expected transform validation", "required": False, "ready": True, "summary": "default moments tolerance"},
        ],
        "ready": True,
        "issues": [],
        "actions": [],
        "summary": "runtime moments | requested moments | anchors off | datum roi_top_left/part_axis",
    }


def _build_runtime_paths(tmp_path) -> dict:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "camera_config.json"
    config_file.write_text(json.dumps({"inspection": {}}, indent=2) + "\n", encoding="utf-8")
    return {
        "config_file": config_file,
        "reference_dir": reference_dir,
        "reference_mask": reference_dir / "reference_mask.png",
        "reference_image": reference_dir / "golden_reference_image.png",
        "log_dir": tmp_path / "logs",
    }


def _write_training_records(active_paths: dict, records: list[dict]) -> None:
    training_file = Path(active_paths["config_file"]).parent / "training_data.json"
    training_file.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def test_ml_mode_warns_when_model_and_threshold_are_missing() -> None:
    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "mask_and_ml"}},
        anomaly_detector=None,
    )

    assert len(warnings) == 2
    assert "no trained anomaly model" in warnings[0].lower()
    assert "min anomaly score" in warnings[1].lower()


def test_full_mode_warns_only_for_missing_threshold_when_model_exists() -> None:
    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "full", "min_anomaly_score": None}},
        anomaly_detector=object(),
    )

    assert warnings == [
        "ML-backed mode is selected but Min Anomaly Score is not set. The anomaly gate is inactive."
    ]


def test_mask_only_mode_has_no_ml_warnings() -> None:
    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "mask_only"}},
        anomaly_detector=None,
    )

    assert warnings == []


def test_describe_edge_gate_status_reports_missing_thresholds() -> None:
    status_line, hint = describe_edge_gate_status({"inspection": {}})

    assert status_line == "Edge Gates: global off | section off"
    assert "set Max Mean Edge Distance and Max Section Edge Distance" in hint


def test_format_operator_mode_lines_includes_edge_gate_hint() -> None:
    lines = format_operator_mode_lines(
        {
            "inspection": {
                "inspection_mode": "mask_only",
                "reference_strategy": "golden_only",
                "blend_mode": "hard_only",
                "tolerance_mode": "balanced",
                "max_mean_edge_distance_px": 1.25,
                "max_section_edge_distance_px": None,
                "max_section_width_delta_ratio": None,
                "max_section_center_offset_px": None,
            }
        }
    )

    assert "Edge Gates: global<=1.25px | section off" in lines
    assert any("set Max Section Edge Distance" in line for line in lines)
    assert "Width Gate: section off" in lines
    assert any("set Max Section Width Drift" in line for line in lines)
    assert "Center Gate: section off" in lines
    assert any("set Max Section Center Offset" in line for line in lines)


def test_describe_section_width_gate_status_reports_missing_threshold() -> None:
    status_line, hint = describe_section_width_gate_status({"inspection": {}})

    assert status_line == "Width Gate: section off"
    assert "set Max Section Width Drift" in hint


def test_describe_section_center_gate_status_reports_missing_threshold() -> None:
    status_line, hint = describe_section_center_gate_status({"inspection": {}})

    assert status_line == "Center Gate: section off"
    assert "set Max Section Center Offset" in hint


def test_ml_mode_warns_when_committed_good_samples_are_insufficient(tmp_path) -> None:
    reference_dir = tmp_path / "reference"
    active_dir = reference_dir / "anomaly_good_library" / "active"
    for index in range(3):
        sample_dir = active_dir / f"sample_{index}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample_meta.json").write_text(
            json.dumps({"sample_asset": {"sample_id": f"sample_{index}"}, "features": [0.1, 0.2, 0.3]}),
            encoding="utf-8",
        )

    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "mask_and_ml", "min_anomaly_score": 0.1}},
        anomaly_detector=None,
        active_paths={"reference_dir": reference_dir},
    )

    assert len(warnings) == 1
    assert "not enough approved-good samples" in warnings[0].lower()
    assert "3/8" in warnings[0]


def test_ml_mode_warns_when_model_is_stale_for_current_sample_library(tmp_path) -> None:
    reference_dir = tmp_path / "reference"
    active_dir = reference_dir / "anomaly_good_library" / "active"
    for index in range(8):
        sample_dir = active_dir / f"sample_{index}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample_meta.json").write_text(
            json.dumps({"sample_asset": {"sample_id": f"sample_{index}"}, "features": [0.1, 0.2, 0.3]}),
            encoding="utf-8",
        )

    (reference_dir / "anomaly_model.pkl").write_bytes(b"model")
    (reference_dir / "anomaly_model_meta.json").write_text(
        json.dumps({"trained_sample_count": 6}),
        encoding="utf-8",
    )

    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "full", "min_anomaly_score": 0.1}},
        anomaly_detector=object(),
        active_paths={"reference_dir": reference_dir},
    )

    assert warnings == [
        "ML-backed mode is selected but the anomaly model is stale for the current approved-good sample library. Press Update in training to rebuild it."
    ]


def test_get_commissioning_status_tracks_hybrid_progress_and_pending_update_actions(tmp_path) -> None:
    active_paths = _build_runtime_paths(tmp_path)
    active_paths["reference_mask"].write_bytes(b"mask")
    active_paths["reference_image"].write_bytes(b"image")
    (active_paths["reference_dir"] / "ref_meta.json").write_text(
        json.dumps(
            {
                "registration_baseline": _default_registration_baseline()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    active_variant_root = active_paths["reference_dir"] / "reference_variants" / "active"
    pending_variant_root = active_paths["reference_dir"] / "reference_variants" / "pending"
    (active_variant_root / "good_a").mkdir(parents=True, exist_ok=True)
    (pending_variant_root / "good_b").mkdir(parents=True, exist_ok=True)
    (pending_variant_root / "good_c").mkdir(parents=True, exist_ok=True)

    _write_training_records(
        active_paths,
        [
            {"feedback": "approve", "final_class": "good", "learning_state": "committed"},
            {"feedback": "approve", "final_class": "good", "learning_state": "committed"},
            {"feedback": "approve", "final_class": "good", "learning_state": "committed"},
            {"feedback": "approve", "final_class": "good", "learning_state": "committed"},
            {"feedback": "approve", "final_class": "good", "learning_state": "pending"},
            {"feedback": "approve", "final_class": "good", "learning_state": "pending"},
        ],
    )

    status = get_commissioning_status(
        {
            "inspection": {"reference_strategy": "hybrid", "inspection_mode": "mask_only"},
            "training": {"hybrid_min_good_samples": 6, "hybrid_min_active_variants": 3},
        },
        active_paths,
    )

    assert status["ready"] is False
    assert "Commissioning: NOT READY" in status["summary_line"]
    assert status["workflow_stage_title"] == "Commit Baseline"
    assert status["workflow_instruction"] == "Baseline captured. Press Update to commit 2 pending approved-good parts."
    assert status["registration_summary"] == "runtime moments | requested moments | anchors off | datum roi_top_left/part_axis"
    assert "golden ok" in status["summary_line"]
    assert "reg ok" in status["summary_line"]
    assert "baseline ok" in status["summary_line"]
    assert "good 4/6" in status["summary_line"]
    assert "refs 1/3" in status["summary_line"]
    assert "Press Update to commit 2 pending approved-good parts." in status["actions"]
    assert "Press Update to activate 2 pending approved-good references." in status["actions"]


def test_mask_only_mode_warns_when_golden_only_commissioning_is_incomplete(tmp_path) -> None:
    active_paths = _build_runtime_paths(tmp_path)
    active_paths["reference_mask"].write_bytes(b"mask")
    active_paths["reference_image"].write_bytes(b"image")

    _write_training_records(
        active_paths,
        [
            {"feedback": "approve", "final_class": "good", "learning_state": "committed"},
            {"feedback": "approve", "final_class": "good", "learning_state": "committed"},
            {"feedback": "approve", "final_class": "good", "learning_state": "committed"},
        ],
    )

    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "mask_only", "reference_strategy": "golden_only"}},
        anomaly_detector=None,
        active_paths=active_paths,
    )

    assert warnings == [
        "Commissioning is incomplete for golden_only: registration baseline not captured; approved-good baseline 3/10. Collect 7 more approved-good parts."
    ]


def test_format_operator_mode_lines_include_commissioning_summary_and_actions(tmp_path) -> None:
    active_paths = _build_runtime_paths(tmp_path)
    active_paths["reference_mask"].write_bytes(b"mask")
    active_paths["reference_image"].write_bytes(b"image")
    (active_paths["reference_dir"] / "ref_meta.json").write_text(
        json.dumps(
            {
                "registration_baseline": _default_registration_baseline()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_training_records(
        active_paths,
        [
            {"feedback": "approve", "final_class": "good", "learning_state": "committed"},
            {"feedback": "approve", "final_class": "good", "learning_state": "committed"},
        ],
    )

    lines = format_operator_mode_lines(
        {"inspection": {"inspection_mode": "mask_only", "reference_strategy": "golden_only"}},
        active_paths=active_paths,
    )

    assert "Workflow: Stage 3/5 - Baseline Build" in lines
    assert "Instruction: Load known-good part. More good examples needed: 8 of 10 remaining." in lines
    assert "Registration: runtime moments | requested moments | anchors off | datum roi_top_left/part_axis" in lines
    assert any(line.startswith("Commissioning: NOT READY | golden ok | reg ok | baseline ok | good 2/10") for line in lines)
    assert "Next: Collect 8 more approved-good parts." in lines


def test_get_commissioning_status_surfaces_registration_setup_stage(tmp_path) -> None:
    active_paths = _build_runtime_paths(tmp_path)
    active_paths["reference_mask"].write_bytes(b"mask")
    active_paths["reference_image"].write_bytes(b"image")
    (active_paths["reference_dir"] / "ref_meta.json").write_text(
        json.dumps(
            {
                "registration_baseline": {
                    "ready": False,
                    "summary": "runtime anchor_pair | requested anchor_pair | anchors 1/2 | datum anchor_primary/anchor_pair | datum pending | transform pending",
                    "issues": [
                        "enabled anchors 1/2 for pair registration",
                        "search windows missing for left_pad",
                        "datum frame not confirmed",
                        "expected transform limits not validated",
                    ],
                    "actions": [
                        "Define at least 2 enabled registration anchors before relying on anchor_pair.",
                        "Define search windows for the enabled registration anchors.",
                        "Confirm the datum frame in Registration Setup after reviewing the anchor-origin/orientation mapping.",
                        "Review the expected angle/shift limits and confirm them in Registration Setup.",
                    ],
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    status = get_commissioning_status(
        {
            "inspection": {"reference_strategy": "golden_only", "inspection_mode": "mask_only"},
            "alignment": {
                "mode": "anchor_pair",
                "registration": {
                    "strategy": "anchor_pair",
                    "anchor_mode": "pair",
                    "datum_frame": {"origin": "anchor_primary", "orientation": "anchor_pair"},
                    "anchors": [
                        {
                            "anchor_id": "left_pad",
                            "enabled": True,
                            "reference_point": {"x": 10, "y": 12},
                            "search_window": {"x": 0, "y": 0, "width": 0, "height": 0},
                        }
                    ],
                },
            },
            "training": {"golden_only_min_good_samples": 10},
        },
        active_paths,
    )

    assert status["registration_ready"] is False
    assert status["workflow_stage_title"] == "Registration Setup"
    assert status["workflow_instruction"] == "Define at least 2 enabled registration anchors before relying on anchor_pair."
    assert "reg setup" in status["summary_line"]
    assert "registration setup incomplete" in status["warning"]


def test_get_commissioning_status_marks_hybrid_activation_prompt_when_golden_only_is_ready(tmp_path) -> None:
    active_paths = _build_runtime_paths(tmp_path)
    active_paths["reference_mask"].write_bytes(b"mask")
    active_paths["reference_image"].write_bytes(b"image")
    (active_paths["reference_dir"] / "ref_meta.json").write_text(
        json.dumps({"registration_baseline": _default_registration_baseline()}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_training_records(
        active_paths,
        [{"feedback": "approve", "final_class": "good", "learning_state": "committed"} for _ in range(10)],
    )

    status = get_commissioning_status(
        {"inspection": {"inspection_mode": "mask_only", "reference_strategy": "golden_only"}},
        active_paths=active_paths,
    )

    assert status["ready"] is True
    assert status["workflow_stage_title"] == "Production Ready"
    assert status["workflow_upgrade_prompt"] == (
        "Hybrid now available. Activate if molded-part variation needs multiple approved-good references."
    )


def test_print_inspection_result_calls_out_registration_rejection() -> None:
    details = {
        "inspection_mode": "mask_only",
        "registration": {
            "status": "quality_gate_failed",
            "applied_strategy": "anchor_translation",
            "rejection_reason": "Registration confidence 0.500 was below required minimum 0.900.",
            "quality_gate_failures": [
                {
                    "summary": "Registration confidence 0.500 was below required minimum 0.900.",
                }
            ],
        },
        "failure_stage": "registration",
        "roi": {"x": 0, "y": 0, "width": 20, "height": 20},
        "best_angle_deg": 0.0,
        "best_shift_x": 0,
        "best_shift_y": 0,
        "required_coverage": 0.95,
        "min_required_coverage": 0.92,
        "outside_allowed_ratio": 0.01,
        "max_outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.90,
        "min_section_coverage_limit": 0.85,
        "sample_white_pixels": 42,
    }

    with mock.patch("builtins.print") as print_mock:
        print_inspection_result(False, details)

    printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
    assert "Registration: quality_gate_failed via anchor_translation" in printed
    assert "Registration rejection: Registration confidence 0.500 was below required minimum 0.900." in printed
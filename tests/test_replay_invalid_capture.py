from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import replay_inspection
from inspection_system.app.result_status import PASS, REGISTRATION_FAILED


class FakeCv2:
    IMREAD_COLOR = 1

    def __init__(self, image):
        self._image = image

    def imread(self, _path: str, _mode: int):
        return self._image


def test_classify_invalid_capture_reports_missing_image(monkeypatch, tmp_path: Path) -> None:
    fake_cv2 = FakeCv2(image=None)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    missing = tmp_path / "missing.jpg"
    reason = replay_inspection.classify_invalid_capture({"inspection": {"roi": {}}}, missing)

    assert "does not exist" in reason


def test_classify_invalid_capture_reports_bad_roi(monkeypatch, tmp_path: Path) -> None:
    fake_image = SimpleNamespace(shape=(100, 200, 3))
    fake_cv2 = FakeCv2(image=fake_image)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(replay_inspection, "REFERENCE_MASK", tmp_path / "golden_reference_mask.png")
    replay_inspection.REFERENCE_MASK.write_text("placeholder", encoding="utf-8")

    image_path = tmp_path / "sample.jpg"
    image_path.write_text("placeholder", encoding="utf-8")

    reason = replay_inspection.classify_invalid_capture(
        {
            "inspection": {
                "roi": {"x": 180, "y": 10, "width": 40, "height": 20}
            }
        },
        image_path,
    )

    assert reason == "Configured ROI is outside image bounds."


def test_classify_invalid_capture_reports_missing_reference(monkeypatch, tmp_path: Path) -> None:
    fake_image = SimpleNamespace(shape=(100, 200, 3))
    fake_cv2 = FakeCv2(image=fake_image)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(replay_inspection, "REFERENCE_MASK", tmp_path / "missing_reference.png")

    image_path = tmp_path / "sample.jpg"
    image_path.write_text("placeholder", encoding="utf-8")

    reason = replay_inspection.classify_invalid_capture({"inspection": {"roi": {}}}, image_path)

    assert "Reference mask is missing" in reason


def test_inspect_file_uses_runtime_reference_paths_and_anomaly_detector(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_text("placeholder", encoding="utf-8")

    active_paths = {
        "reference_mask": tmp_path / "runtime_mask.png",
        "reference_image": tmp_path / "runtime_image.png",
        "reference_dir": tmp_path,
    }
    detector = object()
    captured = {}

    monkeypatch.setattr(replay_inspection, "classify_invalid_capture", lambda config, path, active_paths=None: None)
    monkeypatch.setattr(replay_inspection, "get_active_runtime_paths", lambda: active_paths)
    monkeypatch.setattr(replay_inspection, "load_anomaly_detector", lambda paths: detector)
    monkeypatch.setattr(
        replay_inspection,
        "list_runtime_reference_candidates",
        lambda config, paths: [
            {
                "reference_id": "golden",
                "label": "Golden Reference",
                "role": "golden",
                "reference_mask_path": active_paths["reference_mask"],
                "reference_image_path": active_paths["reference_image"],
            }
        ],
    )

    def fake_inspect_against_references(
        config,
        sample_image_path,
        reference_candidates,
        make_binary_mask,
        *args,
        anomaly_detector=None,
    ):
        captured["sample_image_path"] = sample_image_path
        captured["reference_mask_path"] = reference_candidates[0]["reference_mask_path"]
        captured["reference_image_path"] = reference_candidates[0]["reference_image_path"]
        captured["anomaly_detector"] = anomaly_detector
        return True, {
            "reference_id": "golden",
            "reference_label": "Golden Reference",
            "reference_role": "golden",
            "reference_strategy": "hybrid",
            "reference_candidate_count": 1,
            "evaluated_reference_ids": ["golden"],
            "reference_candidate_errors": [],
            "reference_candidate_summaries": [
                {
                    "reference_id": "golden",
                    "reference_label": "Golden Reference",
                    "reference_role": "golden",
                    "passed": True,
                    "rank": {
                        "passed_score": 1,
                        "failed_gate_score": 0,
                        "margin_score": 0.5,
                    },
                    "registration_status": "aligned",
                    "registration_runtime_mode": "anchor_translation",
                    "registration_applied_strategy": "anchor_translation",
                    "registration_datum_frame": "reference_mask",
                    "edge_measurement_frame": "datum",
                    "section_measurement_frame": "datum",
                    "required_coverage": 0.95,
                    "outside_allowed_ratio": 0.01,
                    "min_section_coverage": 0.9,
                    "mean_edge_distance_px": 0.0,
                    "worst_section_edge_distance_px": 0.0,
                    "worst_section_width_delta_ratio": 0.0,
                    "worst_section_center_offset_px": 0.0,
                }
            ],
            "required_coverage": 0.95,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.90,
            "sample_white_pixels": 42,
            "best_angle_deg": 0.0,
            "best_shift_x": 0,
            "best_shift_y": 0,
            "registration": {
                "status": "aligned",
                "runtime_mode": "anchor_translation",
                "applied_strategy": "anchor_translation",
                "datum_frame": "reference_mask",
            },
            "edge_measurement_frame": "datum",
            "section_measurement_frame": "datum",
        }

    monkeypatch.setattr(replay_inspection, "inspect_against_references", fake_inspect_against_references)

    result = replay_inspection.inspect_file({"inspection": {}}, image_path, active_paths=active_paths)

    assert result["status"] == PASS
    assert captured["sample_image_path"] == image_path
    assert captured["reference_mask_path"] == active_paths["reference_mask"]
    assert captured["reference_image_path"] == active_paths["reference_image"]
    assert captured["anomaly_detector"] is detector
    assert result["reference_id"] == "golden"
    assert result["reference_label"] == "Golden Reference"
    assert result["reference_role"] == "golden"
    assert result["reference_strategy"] == "hybrid"
    assert result["reference_candidate_count"] == 1
    assert result["evaluated_reference_ids"] == ["golden"]
    assert result["reference_candidate_errors"] == []
    assert result["reference_candidate_summaries"][0]["reference_id"] == "golden"
    assert result["reference_candidate_summaries"][0]["rank"]["margin_score"] == 0.5
    assert result["registration_status"] == "aligned"
    assert result["registration_runtime_mode"] == "anchor_translation"
    assert result["registration_applied_strategy"] == "anchor_translation"
    assert result["registration_datum_frame"] == "reference_mask"
    assert result["edge_measurement_frame"] == "datum"
    assert result["section_measurement_frame"] == "datum"
    assert result["inspection_mode"] == "mask_only"


def test_inspect_file_returns_registration_failed_when_registration_is_rejected(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_text("placeholder", encoding="utf-8")

    active_paths = {
        "reference_mask": tmp_path / "runtime_mask.png",
        "reference_image": tmp_path / "runtime_image.png",
        "reference_dir": tmp_path,
    }

    monkeypatch.setattr(replay_inspection, "classify_invalid_capture", lambda config, path, active_paths=None: None)
    monkeypatch.setattr(replay_inspection, "get_active_runtime_paths", lambda: active_paths)
    monkeypatch.setattr(replay_inspection, "load_anomaly_detector", lambda paths: None)
    monkeypatch.setattr(
        replay_inspection,
        "list_runtime_reference_candidates",
        lambda config, paths: [
            {
                "reference_id": "golden",
                "label": "Golden Reference",
                "role": "golden",
                "reference_mask_path": active_paths["reference_mask"],
                "reference_image_path": active_paths["reference_image"],
            }
        ],
    )
    monkeypatch.setattr(
        replay_inspection,
        "inspect_against_references",
        lambda *args, **kwargs: (
            False,
            {
                "reference_id": "golden",
                "reference_label": "Golden Reference",
                "reference_role": "golden",
                "reference_strategy": "hybrid",
                "reference_candidate_count": 1,
                "evaluated_reference_ids": ["golden"],
                "reference_candidate_errors": [],
                "reference_candidate_summaries": [],
                "required_coverage": 0.95,
                "outside_allowed_ratio": 0.01,
                "min_section_coverage": 0.9,
                "sample_white_pixels": 42,
                "best_angle_deg": 0.0,
                "best_shift_x": 0,
                "best_shift_y": 0,
                "registration": {
                    "status": "quality_gate_failed",
                    "runtime_mode": "anchor_translation",
                    "applied_strategy": "anchor_translation",
                    "datum_frame": "reference_mask",
                    "rejection_reason": "Registration confidence 0.500 was below required minimum 0.900.",
                    "quality_gate_failures": [
                        {
                            "cause_code": "registration_failure",
                            "gate_key": "min_confidence",
                            "summary": "Registration confidence 0.500 was below required minimum 0.900.",
                        }
                    ],
                },
                "failure_stage": "registration",
                "edge_measurement_frame": "aligned_mask",
                "section_measurement_frame": "aligned_mask",
            },
        ),
    )

    result = replay_inspection.inspect_file({"inspection": {}}, image_path, active_paths=active_paths)

    assert result["status"] == REGISTRATION_FAILED
    assert result["registration_rejected"] is True
    assert result["registration_rejection_reason"] == "Registration confidence 0.500 was below required minimum 0.900."
    assert result["failure_stage"] == "registration"


def test_inspect_file_serializes_named_feature_position_summary(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_text("placeholder", encoding="utf-8")

    active_paths = {
        "reference_mask": tmp_path / "runtime_mask.png",
        "reference_image": tmp_path / "runtime_image.png",
        "reference_dir": tmp_path,
    }

    monkeypatch.setattr(replay_inspection, "classify_invalid_capture", lambda config, path, active_paths=None: None)
    monkeypatch.setattr(replay_inspection, "get_active_runtime_paths", lambda: active_paths)
    monkeypatch.setattr(replay_inspection, "load_anomaly_detector", lambda paths: None)
    monkeypatch.setattr(
        replay_inspection,
        "list_runtime_reference_candidates",
        lambda config, paths: [
            {
                "reference_id": "golden",
                "label": "Golden Reference",
                "role": "golden",
                "reference_mask_path": active_paths["reference_mask"],
                "reference_image_path": active_paths["reference_image"],
            }
        ],
    )
    monkeypatch.setattr(
        replay_inspection,
        "inspect_against_references",
        lambda *args, **kwargs: (
            False,
            {
                "reference_id": "golden",
                "reference_label": "Golden Reference",
                "reference_role": "golden",
                "reference_strategy": "hybrid",
                "reference_candidate_count": 1,
                "evaluated_reference_ids": ["golden"],
                "reference_candidate_errors": [],
                "reference_candidate_summaries": [],
                "required_coverage": 0.95,
                "outside_allowed_ratio": 0.01,
                "min_section_coverage": 0.9,
                "sample_white_pixels": 42,
                "best_angle_deg": 0.0,
                "best_shift_x": 0,
                "best_shift_y": 0,
                "registration": {
                    "status": "aligned",
                    "runtime_mode": "anchor_translation",
                    "applied_strategy": "anchor_translation",
                    "datum_frame": "reference_mask",
                },
                "failure_stage": "inspection",
                "edge_measurement_frame": "datum",
                "section_measurement_frame": "datum",
                "feature_position_summary": {
                    "feature_key": "paired_feature_1",
                    "feature_label": "Paired Feature 1",
                    "feature_family": "paired_centroid",
                    "feature_type": "paired_centroid_position",
                    "measurement_frame": "datum",
                    "feature_count": 1,
                    "sample_detected": True,
                    "failure_cause": "feature_position",
                    "reference_center": {"x": 80.0, "y": 64.0},
                    "observed_center_reference": {"x": 83.0, "y": 63.0},
                    "dx_px": 3.0,
                    "dy_px": -1.0,
                    "radial_offset_px": 3.162278,
                    "center_offset_px": 3.0,
                    "pair_spacing_reference_px": 18.0,
                    "pair_spacing_observed_px": 18.75,
                    "pair_spacing_delta_px": 0.75,
                },
            },
        ),
    )

    result = replay_inspection.inspect_file({"inspection": {}}, image_path, active_paths=active_paths)

    assert result["status"] == replay_inspection.FAIL
    assert result["inspection_failure_cause"] == "feature_position"
    assert result["feature_position_summary"]["feature_key"] == "paired_feature_1"
    assert result["feature_position_summary"]["dx_px"] == 3.0
    assert result["feature_position_summary"]["dy_px"] == -1.0
    assert result["feature_position_summary"]["pair_spacing_delta_px"] == 0.75


def test_inspect_file_serializes_lane_aware_replay_fields(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_text("placeholder", encoding="utf-8")

    active_paths = {
        "reference_mask": tmp_path / "runtime_mask.png",
        "reference_image": tmp_path / "runtime_image.png",
        "reference_dir": tmp_path,
    }

    monkeypatch.setattr(replay_inspection, "classify_invalid_capture", lambda config, path, active_paths=None: None)
    monkeypatch.setattr(replay_inspection, "get_active_runtime_paths", lambda: active_paths)
    monkeypatch.setattr(replay_inspection, "load_anomaly_detector", lambda paths: None)
    monkeypatch.setattr(
        replay_inspection,
        "list_runtime_reference_candidates",
        lambda config, paths: [
            {
                "reference_id": "golden",
                "label": "Golden Reference",
                "role": "golden",
                "reference_mask_path": active_paths["reference_mask"],
                "reference_image_path": active_paths["reference_image"],
            }
        ],
    )
    monkeypatch.setattr(
        replay_inspection,
        "inspect_against_references",
        lambda *args, **kwargs: (
            False,
            {
                "inspection_program": {
                    "program_id": "pilot_program",
                    "aggregation_policy": "all_authoritative",
                    "lane_ids": ["geometry", "print"],
                    "primary_lane_id": "geometry",
                    "active_lane_id": "print",
                },
                "lane_results": [
                    {
                        "lane_id": "geometry",
                        "lane_type": "measurement",
                        "authoritative": True,
                        "passed": True,
                        "threshold_summary": {
                            "required_coverage": 0.98,
                            "outside_allowed_ratio": 0.0,
                            "min_section_coverage": 0.97,
                        },
                    },
                    {
                        "lane_id": "print",
                        "lane_type": "measurement",
                        "authoritative": True,
                        "passed": False,
                        "inspection_failure_cause": "feature_position",
                        "threshold_summary": {
                            "required_coverage": 0.82,
                            "outside_allowed_ratio": 0.0,
                            "min_section_coverage": 0.81,
                        },
                        "feature_position_summary": {
                            "feature_key": "print_1",
                            "feature_label": "Print 1",
                            "failure_cause": "feature_position",
                        },
                    },
                ],
                "failed_lane_ids": ["print"],
                "failed_authoritative_lane_ids": ["print"],
                "failed_advisory_lane_ids": [],
                "reference_id": "golden",
                "reference_label": "Golden Reference",
                "reference_role": "golden",
                "reference_strategy": "hybrid",
                "reference_candidate_count": 1,
                "evaluated_reference_ids": ["golden"],
                "reference_candidate_errors": [],
                "reference_candidate_summaries": [],
                "required_coverage": 0.82,
                "outside_allowed_ratio": 0.0,
                "min_section_coverage": 0.81,
                "sample_white_pixels": 42,
                "best_angle_deg": 0.0,
                "best_shift_x": 0,
                "best_shift_y": 0,
                "registration": {
                    "status": "aligned",
                    "runtime_mode": "moments",
                    "applied_strategy": "moments",
                    "datum_frame": "reference_mask",
                },
                "failure_stage": "inspection",
                "inspection_failure_cause": "feature_position",
                "edge_measurement_frame": "datum",
                "section_measurement_frame": "datum",
            },
        ),
    )

    result = replay_inspection.inspect_file({"inspection": {}}, image_path, active_paths=active_paths)

    assert result["inspection_program"]["program_id"] == "pilot_program"
    assert result["inspection_program"]["active_lane_id"] == "print"
    assert result["failed_lane_ids"] == ["print"]
    assert result["failed_authoritative_lane_ids"] == ["print"]
    assert result["lane_results"][1]["lane_id"] == "print"
    assert result["lane_results"][1]["inspection_failure_cause"] == "feature_position"

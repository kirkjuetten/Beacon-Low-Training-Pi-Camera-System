from __future__ import annotations

from inspection_system.app.inspection_models import (
    OUTCOME_KIND_CONFIG_ERROR,
    OUTCOME_KIND_INVALID_CAPTURE,
    OUTCOME_KIND_REGISTRATION_FAILURE,
    GateDecision,
    InspectionOutcome,
    LaneResult,
    MeasurementBundle,
    RegistrationAssessment,
    infer_outcome_kind,
)


class _RegistrationStub:
    enabled = True
    status = "aligned"
    runtime_mode = "anchor_pair"
    requested_strategy = "anchor_pair"
    applied_strategy = "anchor_pair"
    transform_model = "rigid"
    anchor_mode = "pair"
    subpixel_refinement = "template"
    fallback_reason = None
    rejection_reason = None
    quality = {"confidence": 0.92, "mean_residual_px": 0.3}
    quality_gates = {"min_confidence": 0.8, "max_mean_residual_px": 1.0}
    quality_gate_failures = []
    datum_frame = {"origin": "anchor_primary"}
    transform = {"angle_deg": 0.1, "shift_x": 1, "shift_y": -1}
    observed_anchors = [{"anchor_id": "a"}]


def test_gate_decision_round_trips_legacy_summary() -> None:
    decision = GateDecision.from_legacy(True, {"required_coverage": 0.95, "inspection_mode": "mask_only"})

    assert decision.passed is True
    assert decision.to_legacy_summary() == {"required_coverage": 0.95, "inspection_mode": "mask_only"}


def test_inspection_outcome_round_trips_legacy_details() -> None:
    registration = RegistrationAssessment.from_registration_result(
        _RegistrationStub(),
        scoring_guard_reason="guard",
    )
    measurements = MeasurementBundle.from_legacy(
        {
            "passed": True,
            "threshold_summary": {"required_coverage": 0.95},
            "metrics": {"section_coverages": [0.95]},
            "mean_edge_distance_px": 0.2,
            "section_edge_distances_px": [0.2],
            "worst_section_edge_distance_px": 0.2,
            "section_width_ratios": [1.0],
            "section_center_offsets_px": [0.1],
            "section_measurement_frame": "datum",
            "section_measurements": [{"sample_detected": True}],
            "feature_measurements": [{"feature_key": "f1"}],
            "feature_position_summary": {"feature_key": "f1"},
            "worst_section_width_delta_ratio": 0.0,
            "worst_section_center_offset_px": 0.1,
            "edge_measurement_frame": "datum",
        }
    )
    decision = GateDecision.from_legacy(True, {"required_coverage": 0.95})
    outcome = InspectionOutcome.from_legacy_details(
        passed=True,
        registration=registration,
        measurements=measurements,
        gate_decision=decision,
        details={
            "registration": {},
            "feature_measurements": [],
            "feature_position_summary": None,
            "required_coverage": 0.95,
        },
    )

    legacy = outcome.to_legacy_details()

    assert outcome.outcome_kind == "pass"
    assert legacy["outcome_kind"] == "pass"
    assert legacy["registration"]["runtime_mode"] == "anchor_pair"
    assert legacy["registration"]["scoring_guard_reason"] == "guard"
    assert legacy["feature_measurements"][0]["feature_key"] == "f1"
    assert legacy["feature_position_summary"]["feature_key"] == "f1"


def test_infer_outcome_kind_recognizes_failure_stages() -> None:
    assert infer_outcome_kind(passed=False, details={"failure_stage": "registration"}) == OUTCOME_KIND_REGISTRATION_FAILURE
    assert infer_outcome_kind(passed=False, details={"failure_stage": "invalid_capture"}) == OUTCOME_KIND_INVALID_CAPTURE
    assert infer_outcome_kind(passed=False, details={"failure_stage": "config"}) == OUTCOME_KIND_CONFIG_ERROR
    assert infer_outcome_kind(passed=False, details={"failure_stage": "inspection"}) == "inspection_failure"


def test_lane_result_round_trips_legacy_dict() -> None:
    lane_result = LaneResult.from_legacy(
        {
            "lane_id": "geometry",
            "lane_type": "measurement",
            "authoritative": True,
            "passed": False,
            "inspection_cfg": {"lane_tag": "geometry"},
            "measurement_result": {"mean_edge_distance_px": 0.4},
            "threshold_summary": {"required_coverage": 0.9},
            "feature_measurements": [{"feature_key": "f1"}],
            "feature_position_summary": {"feature_key": "f1"},
            "edge_measurement_frame": "datum",
            "section_measurement_frame": "datum",
            "inspection_failure_cause": "feature_position",
        }
    )

    assert lane_result.lane_id == "geometry"
    assert lane_result.feature_position_summary == {"feature_key": "f1"}
    assert lane_result.to_legacy_dict()["inspection_failure_cause"] == "feature_position"


def test_lane_result_with_failure_cause_returns_updated_copy() -> None:
    lane_result = LaneResult.from_legacy(
        {
            "lane_id": "print",
            "lane_type": "measurement",
            "authoritative": False,
            "passed": True,
        }
    )

    updated = lane_result.with_inspection_failure_cause("coverage")

    assert lane_result.inspection_failure_cause is None
    assert updated.inspection_failure_cause == "coverage"
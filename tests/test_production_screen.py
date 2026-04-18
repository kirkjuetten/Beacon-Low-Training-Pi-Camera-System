from inspection_system.app.production_screen import (
    CounterScope,
)
from inspection_system.app.result_interpreter import (
    GOOD,
    REASON_EXTRA_PRINT,
    REASON_MISSING_PRINT,
    REASON_REGISTRATION_FAILURE,
    REASON_REFERENCE_MISMATCH,
    REASON_UNEVEN_PRINT,
    REJECT,
    REVIEW,
    determine_operator_outcome,
)


def _base_details() -> dict:
    return {
        "required_coverage": 0.98,
        "min_required_coverage": 0.92,
        "outside_allowed_ratio": 0.002,
        "max_outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.94,
        "min_section_coverage_limit": 0.85,
        "edge_distance_gate_active": False,
        "section_width_gate_active": False,
        "section_center_gate_active": False,
        "ssim_gate_active": False,
        "mse_gate_active": False,
        "anomaly_gate_active": False,
    }


def test_determine_operator_outcome_marks_good_parts_good() -> None:
    outcome = determine_operator_outcome(True, _base_details())

    assert outcome.status == GOOD
    assert outcome.banner_text == "GOOD"
    assert outcome.primary_reason is None


def test_determine_operator_outcome_uses_missing_print_precedence() -> None:
    details = _base_details()
    details.update(
        {
            "required_coverage": 0.74,
            "outside_allowed_ratio": 0.18,
            "min_section_coverage": 0.40,
        }
    )

    outcome = determine_operator_outcome(False, details)

    assert outcome.status == REJECT
    assert outcome.primary_reason == REASON_MISSING_PRINT


def test_determine_operator_outcome_routes_borderline_single_failure_to_review() -> None:
    details = _base_details()
    details.update({"required_coverage": 0.89})

    outcome = determine_operator_outcome(False, details)

    assert outcome.status == REVIEW
    assert outcome.primary_reason == REASON_MISSING_PRINT


def test_determine_operator_outcome_uses_extra_print_reason() -> None:
    details = _base_details()
    details.update({"outside_allowed_ratio": 0.08})

    outcome = determine_operator_outcome(False, details)

    assert outcome.status == REJECT
    assert outcome.primary_reason == REASON_EXTRA_PRINT


def test_determine_operator_outcome_uses_uneven_print_reason() -> None:
    details = _base_details()
    details.update({"min_section_coverage": 0.42})

    outcome = determine_operator_outcome(False, details)

    assert outcome.status == REJECT
    assert outcome.primary_reason == REASON_UNEVEN_PRINT


def test_determine_operator_outcome_uses_reference_mismatch_when_gate_fails() -> None:
    details = _base_details()
    details.update(
        {
            "ssim_gate_active": True,
            "min_ssim": 0.90,
            "ssim": 0.72,
        }
    )

    outcome = determine_operator_outcome(False, details)

    assert outcome.status == REJECT
    assert outcome.primary_reason == REASON_REFERENCE_MISMATCH


def test_determine_operator_outcome_uses_reference_mismatch_for_edge_distance_gate() -> None:
    details = _base_details()
    details.update(
        {
            "edge_distance_gate_active": True,
            "mean_edge_distance_px": 1.6,
            "max_mean_edge_distance_px": 1.0,
        }
    )

    outcome = determine_operator_outcome(False, details)

    assert outcome.status == REJECT
    assert outcome.primary_reason == REASON_REFERENCE_MISMATCH


def test_determine_operator_outcome_uses_reference_mismatch_for_section_edge_gate() -> None:
    details = _base_details()
    details.update(
        {
            "section_edge_gate_active": True,
            "worst_section_edge_distance_px": 1.3,
            "max_section_edge_distance_px": 0.8,
        }
    )

    outcome = determine_operator_outcome(False, details)

    assert outcome.status == REJECT
    assert outcome.primary_reason == REASON_REFERENCE_MISMATCH


def test_determine_operator_outcome_uses_reference_mismatch_for_section_width_gate() -> None:
    details = _base_details()
    details.update(
        {
            "section_width_gate_active": True,
            "worst_section_width_delta_ratio": 0.18,
            "max_section_width_delta_ratio": 0.1,
        }
    )

    outcome = determine_operator_outcome(False, details)

    assert outcome.status == REJECT
    assert outcome.primary_reason == REASON_REFERENCE_MISMATCH


def test_determine_operator_outcome_uses_reference_mismatch_for_section_center_gate() -> None:
    details = _base_details()
    details.update(
        {
            "section_center_gate_active": True,
            "worst_section_center_offset_px": 1.2,
            "max_section_center_offset_px": 0.6,
        }
    )

    outcome = determine_operator_outcome(False, details)

    assert outcome.status == REJECT
    assert outcome.primary_reason == REASON_REFERENCE_MISMATCH


def test_determine_operator_outcome_routes_registration_failures_to_reload_review() -> None:
    details = _base_details()
    details.update(
        {
            "failure_stage": "registration",
            "registration": {
                "rejection_reason": "Registration confidence 0.500 was below required minimum 0.900.",
            },
        }
    )

    outcome = determine_operator_outcome(False, details)

    assert outcome.status == REVIEW
    assert outcome.banner_text == "CHECK PLACEMENT"
    assert outcome.primary_reason == REASON_REGISTRATION_FAILURE
    assert outcome.summary_lines[0] == "Part position could not be verified."
    assert "Registration confidence 0.500 was below required minimum 0.900." in outcome.summary_lines[1]


def test_counter_scope_records_and_resets() -> None:
    scope = CounterScope()
    good_outcome = determine_operator_outcome(True, _base_details())
    review_outcome = determine_operator_outcome(False, {**_base_details(), "required_coverage": 0.89})
    reject_outcome = determine_operator_outcome(False, {**_base_details(), "outside_allowed_ratio": 0.09})

    scope.record(good_outcome)
    scope.record(review_outcome)
    scope.record(reject_outcome)

    assert scope.total == 3
    assert scope.good == 1
    assert scope.review == 1
    assert scope.reject == 1
    assert scope.reject_reasons[REASON_EXTRA_PRINT] == 1

    scope.reset()

    assert scope.total == 0
    assert scope.good == 0
    assert scope.review == 0
    assert scope.reject == 0
    assert all(count == 0 for count in scope.reject_reasons.values())
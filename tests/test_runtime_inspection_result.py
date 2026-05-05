from pathlib import Path

from inspection_system.app.result_status import CONFIG_ERROR, INVALID_CAPTURE, PASS, REGISTRATION_FAILED
from inspection_system.app.runtime_inspection_result import RuntimeInspectionResult


def test_runtime_inspection_result_from_inspection_uses_typed_failure_family() -> None:
    result = RuntimeInspectionResult.from_inspection(
        Path("sample.jpg"),
        False,
        {
            "failure_stage": "registration",
            "registration": {"rejection_reason": "Registration confidence too low."},
        },
    )

    assert result.status == REGISTRATION_FAILED
    assert result.outcome_kind == "registration_failure"
    assert result.reason == "Registration confidence too low."
    assert result.exit_code == 1


def test_runtime_inspection_result_handles_invalid_capture_and_config_error() -> None:
    invalid_result = RuntimeInspectionResult.from_invalid_capture(Path("sample.jpg"), "ROI is outside bounds.")
    config_result = RuntimeInspectionResult.from_config_error(Path("sample.jpg"), "Reference mask not found.")

    assert invalid_result.status == INVALID_CAPTURE
    assert invalid_result.outcome_kind == "invalid_capture"
    assert config_result.status == CONFIG_ERROR
    assert config_result.outcome_kind == "config_error"


def test_runtime_inspection_result_serializes_base_fields_and_evidence() -> None:
    result = RuntimeInspectionResult.from_inspection(
        Path("sample.jpg"),
        True,
        {"failure_stage": "inspection", "required_coverage": 0.95},
    )

    payload = result.to_legacy_dict(evidence_serializer=lambda evidence: {"required_coverage": evidence["required_coverage"]})

    assert payload == {
        "image": "sample.jpg",
        "status": PASS,
        "outcome_kind": "pass",
        "required_coverage": 0.95,
    }


def test_runtime_inspection_result_reconstructs_from_serialized_result() -> None:
    result = RuntimeInspectionResult.from_serialized_result(
        {
            "image": "sample.jpg",
            "status": REGISTRATION_FAILED,
            "outcome_kind": "registration_failure",
            "reason": "Registration confidence too low.",
            "failure_stage": "registration",
            "registration_rejection_reason": "Registration confidence too low.",
            "registration_quality_gate_failures": [{"gate_key": "min_confidence"}],
        }
    )

    assert result.image_path == "sample.jpg"
    assert result.status == REGISTRATION_FAILED
    assert result.outcome_kind == "registration_failure"
    assert result.registration_rejection_reason == "Registration confidence too low."
    assert result.registration_quality_gate_failures == [{"gate_key": "min_confidence"}]
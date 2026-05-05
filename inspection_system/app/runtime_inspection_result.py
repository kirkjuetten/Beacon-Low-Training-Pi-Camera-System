from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from inspection_system.app.inspection_models import (
    OUTCOME_KIND_CONFIG_ERROR,
    OUTCOME_KIND_INSPECTION_FAILURE,
    OUTCOME_KIND_INVALID_CAPTURE,
    OUTCOME_KIND_PASS,
    OUTCOME_KIND_REGISTRATION_FAILURE,
    infer_outcome_kind,
)
from inspection_system.app.result_status import CONFIG_ERROR, FAIL, INVALID_CAPTURE, PASS, REGISTRATION_FAILED


_SERIALIZED_BASE_KEYS = {"image", "status", "outcome_kind", "reason"}


def resolve_runtime_status(outcome_kind: str, *, passed: bool = False) -> str:
    normalized = str(outcome_kind or "").strip().lower()
    if normalized == OUTCOME_KIND_PASS:
        return PASS
    if normalized == OUTCOME_KIND_REGISTRATION_FAILURE:
        return REGISTRATION_FAILED
    if normalized == OUTCOME_KIND_INVALID_CAPTURE:
        return INVALID_CAPTURE
    if normalized == OUTCOME_KIND_CONFIG_ERROR:
        return CONFIG_ERROR
    if normalized == OUTCOME_KIND_INSPECTION_FAILURE:
        return FAIL
    return PASS if passed else FAIL


def resolve_runtime_reason(*, passed: bool, details: dict) -> str | None:
    if passed:
        return None

    registration = details.get("registration", {}) if isinstance(details.get("registration"), dict) else {}
    rejection_reason = registration.get("rejection_reason") or details.get("registration_rejection_reason")
    if rejection_reason:
        return str(rejection_reason)

    inspection_failure_cause = details.get("inspection_failure_cause")
    if inspection_failure_cause:
        return str(inspection_failure_cause)

    feature_position_summary = details.get("feature_position_summary")
    if isinstance(feature_position_summary, dict) and feature_position_summary.get("failure_cause"):
        return str(feature_position_summary.get("failure_cause"))

    explicit_reason = details.get("reason")
    if explicit_reason:
        return str(explicit_reason)

    return None


@dataclass(frozen=True)
class RuntimeInspectionResult:
    image_path: str
    status: str
    outcome_kind: str
    passed: bool
    reason: str | None
    evidence: dict

    @property
    def exit_code(self) -> int:
        return 0 if self.status == PASS else 1

    @property
    def failure_stage(self) -> str | None:
        value = self.evidence.get("failure_stage")
        return None if value is None else str(value)

    @property
    def registration_rejection_reason(self) -> str | None:
        registration = self.evidence.get("registration", {}) if isinstance(self.evidence.get("registration"), dict) else {}
        value = registration.get("rejection_reason") or self.evidence.get("registration_rejection_reason") or self.reason
        return None if value is None else str(value)

    @property
    def registration_quality_gate_failures(self) -> list[dict]:
        registration = self.evidence.get("registration", {}) if isinstance(self.evidence.get("registration"), dict) else {}
        failures = registration.get("quality_gate_failures") or self.evidence.get("registration_quality_gate_failures") or []
        return deepcopy(list(failures))

    @property
    def inspection_failure_cause(self) -> str | None:
        value = self.evidence.get("inspection_failure_cause")
        if value is None:
            feature_position_summary = self.feature_position_summary
            if isinstance(feature_position_summary, dict) and feature_position_summary.get("failure_cause"):
                value = feature_position_summary.get("failure_cause")
        return None if value is None else str(value)

    @property
    def feature_position_summary(self) -> dict | None:
        summary = self.evidence.get("feature_position_summary")
        return deepcopy(summary) if isinstance(summary, dict) else None

    @classmethod
    def from_inspection(cls, image_path: Path, passed: bool, details: dict) -> "RuntimeInspectionResult":
        outcome_kind = infer_outcome_kind(passed=passed, details=details)
        return cls(
            image_path=str(image_path),
            status=resolve_runtime_status(outcome_kind, passed=passed),
            outcome_kind=outcome_kind,
            passed=bool(passed),
            reason=resolve_runtime_reason(passed=passed, details=details),
            evidence=deepcopy(details),
        )

    @classmethod
    def from_serialized_result(cls, result: dict) -> "RuntimeInspectionResult":
        payload = deepcopy(result)
        image_path = str(payload.pop("image", ""))
        status = str(payload.pop("status", FAIL))
        outcome_kind = str(payload.pop("outcome_kind", "")).strip().lower()
        passed = status == PASS
        reason = payload.pop("reason", None)

        if not outcome_kind:
            outcome_kind = infer_outcome_kind(passed=passed, details=payload)

        return cls(
            image_path=image_path,
            status=status,
            outcome_kind=outcome_kind,
            passed=passed,
            reason=None if reason is None else str(reason),
            evidence=payload,
        )

    @classmethod
    def from_invalid_capture(cls, image_path: Path, reason: str) -> "RuntimeInspectionResult":
        return cls(
            image_path=str(image_path),
            status=INVALID_CAPTURE,
            outcome_kind=OUTCOME_KIND_INVALID_CAPTURE,
            passed=False,
            reason=str(reason),
            evidence={},
        )

    @classmethod
    def from_config_error(cls, image_path: Path, reason: str) -> "RuntimeInspectionResult":
        return cls(
            image_path=str(image_path),
            status=CONFIG_ERROR,
            outcome_kind=OUTCOME_KIND_CONFIG_ERROR,
            passed=False,
            reason=str(reason),
            evidence={},
        )

    def to_legacy_dict(self, *, evidence_serializer: Callable[[dict], dict] | None = None) -> dict:
        payload = {
            "image": self.image_path,
            "status": self.status,
            "outcome_kind": self.outcome_kind,
        }
        if self.reason is not None:
            payload["reason"] = self.reason

        evidence = deepcopy(self.evidence)
        if evidence_serializer is not None:
            evidence = evidence_serializer(evidence)
        payload.update(evidence)
        return payload

    def to_operator_outcome(self):
        from inspection_system.app.result_interpreter import determine_operator_outcome

        return determine_operator_outcome(self.passed, self.evidence)
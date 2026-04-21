from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    summary: dict

    @classmethod
    def from_legacy(cls, passed: bool, summary: Optional[dict]) -> "GateDecision":
        return cls(passed=bool(passed), summary=deepcopy(summary or {}))

    def to_legacy_summary(self) -> dict:
        return deepcopy(self.summary)


@dataclass(frozen=True)
class RegistrationAssessment:
    enabled: bool
    status: str
    runtime_mode: str
    requested_strategy: str
    applied_strategy: str
    transform_model: str
    anchor_mode: str
    subpixel_refinement: str
    fallback_reason: str | None
    rejection_reason: str | None
    quality: dict
    quality_gates: dict
    quality_gate_failures: list[dict]
    datum_frame: dict
    transform: dict
    observed_anchors: list[dict]
    scoring_guard_applied: bool
    scoring_guard_reason: str | None

    @classmethod
    def from_registration_result(
        cls,
        registration_result: Any,
        *,
        transform: Optional[dict] = None,
        scoring_guard_reason: str | None = None,
    ) -> "RegistrationAssessment":
        fallback_reason = registration_result.fallback_reason
        if registration_result.fallback_reason is None and scoring_guard_reason is not None:
            fallback_reason = scoring_guard_reason
        elif registration_result.fallback_reason is not None and scoring_guard_reason is not None:
            fallback_reason = f"{registration_result.fallback_reason} {scoring_guard_reason}"

        return cls(
            enabled=bool(registration_result.enabled),
            status=str(registration_result.status),
            runtime_mode=str(registration_result.runtime_mode),
            requested_strategy=str(registration_result.requested_strategy),
            applied_strategy=str(registration_result.applied_strategy),
            transform_model=str(registration_result.transform_model),
            anchor_mode=str(registration_result.anchor_mode),
            subpixel_refinement=str(registration_result.subpixel_refinement),
            fallback_reason=fallback_reason,
            rejection_reason=registration_result.rejection_reason,
            quality=deepcopy(registration_result.quality),
            quality_gates=deepcopy(registration_result.quality_gates),
            quality_gate_failures=deepcopy(registration_result.quality_gate_failures),
            datum_frame=deepcopy(registration_result.datum_frame),
            transform=deepcopy(transform if transform is not None else registration_result.transform),
            observed_anchors=deepcopy(registration_result.observed_anchors),
            scoring_guard_applied=bool(scoring_guard_reason),
            scoring_guard_reason=scoring_guard_reason,
        )

    def to_legacy_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "runtime_mode": self.runtime_mode,
            "requested_strategy": self.requested_strategy,
            "applied_strategy": self.applied_strategy,
            "transform_model": self.transform_model,
            "anchor_mode": self.anchor_mode,
            "subpixel_refinement": self.subpixel_refinement,
            "fallback_reason": self.fallback_reason,
            "rejection_reason": self.rejection_reason,
            "quality": deepcopy(self.quality),
            "quality_gates": deepcopy(self.quality_gates),
            "quality_gate_failures": deepcopy(self.quality_gate_failures),
            "datum_frame": deepcopy(self.datum_frame),
            "transform": deepcopy(self.transform),
            "observed_anchors": deepcopy(self.observed_anchors),
            "scoring_guard_applied": self.scoring_guard_applied,
            "scoring_guard_reason": self.scoring_guard_reason,
        }


@dataclass(frozen=True)
class MeasurementBundle:
    passed: bool
    threshold_summary: dict
    metrics: dict
    mean_edge_distance_px: Any
    section_edge_distances_px: list
    worst_section_edge_distance_px: Any
    section_width_ratios: list
    section_center_offsets_px: list
    section_measurement_frame: str
    section_measurements: list
    feature_measurements: list[dict]
    feature_position_summary: dict | None
    worst_section_width_delta_ratio: Any
    worst_section_center_offset_px: Any
    edge_measurement_frame: str

    @classmethod
    def from_legacy(cls, measurement_result: dict) -> "MeasurementBundle":
        return cls(
            passed=bool(measurement_result.get("passed")),
            threshold_summary=deepcopy(measurement_result.get("threshold_summary", {})),
            metrics=deepcopy(measurement_result.get("metrics", {})),
            mean_edge_distance_px=measurement_result.get("mean_edge_distance_px"),
            section_edge_distances_px=deepcopy(measurement_result.get("section_edge_distances_px", [])),
            worst_section_edge_distance_px=measurement_result.get("worst_section_edge_distance_px"),
            section_width_ratios=deepcopy(measurement_result.get("section_width_ratios", [])),
            section_center_offsets_px=deepcopy(measurement_result.get("section_center_offsets_px", [])),
            section_measurement_frame=str(measurement_result.get("section_measurement_frame", "aligned_mask")),
            section_measurements=deepcopy(measurement_result.get("section_measurements", [])),
            feature_measurements=deepcopy(measurement_result.get("feature_measurements", [])),
            feature_position_summary=deepcopy(measurement_result.get("feature_position_summary")),
            worst_section_width_delta_ratio=measurement_result.get("worst_section_width_delta_ratio"),
            worst_section_center_offset_px=measurement_result.get("worst_section_center_offset_px"),
            edge_measurement_frame=str(measurement_result.get("edge_measurement_frame", "aligned_mask")),
        )

    def to_legacy_dict(self) -> dict:
        return {
            "passed": self.passed,
            "threshold_summary": deepcopy(self.threshold_summary),
            "metrics": deepcopy(self.metrics),
            "mean_edge_distance_px": self.mean_edge_distance_px,
            "section_edge_distances_px": deepcopy(self.section_edge_distances_px),
            "worst_section_edge_distance_px": self.worst_section_edge_distance_px,
            "section_width_ratios": deepcopy(self.section_width_ratios),
            "section_center_offsets_px": deepcopy(self.section_center_offsets_px),
            "section_measurement_frame": self.section_measurement_frame,
            "section_measurements": deepcopy(self.section_measurements),
            "feature_measurements": deepcopy(self.feature_measurements),
            "feature_position_summary": deepcopy(self.feature_position_summary),
            "worst_section_width_delta_ratio": self.worst_section_width_delta_ratio,
            "worst_section_center_offset_px": self.worst_section_center_offset_px,
            "edge_measurement_frame": self.edge_measurement_frame,
        }


@dataclass(frozen=True)
class InspectionOutcome:
    passed: bool
    registration: RegistrationAssessment
    measurements: MeasurementBundle
    gate_decision: GateDecision
    details: dict

    @classmethod
    def from_legacy_details(
        cls,
        *,
        passed: bool,
        registration: RegistrationAssessment,
        measurements: MeasurementBundle,
        gate_decision: GateDecision,
        details: dict,
    ) -> "InspectionOutcome":
        return cls(
            passed=bool(passed),
            registration=registration,
            measurements=measurements,
            gate_decision=gate_decision,
            details=deepcopy(details),
        )

    def to_legacy_details(self) -> dict:
        legacy = deepcopy(self.details)
        legacy["registration"] = self.registration.to_legacy_dict()
        legacy["feature_measurements"] = deepcopy(self.measurements.feature_measurements)
        legacy["feature_position_summary"] = deepcopy(self.measurements.feature_position_summary)
        return legacy
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import math
from pathlib import Path
from inspection_system.app.anomaly_detection_utils import detect_anomalies
from inspection_system.app.datum_measurement_utils import compute_datum_section_measurements
from inspection_system.app.feature_measurement_utils import (
    DEFAULT_MOLDED_PART_FEATURE_FAMILIES,
    extract_molded_part_feature_measurements,
)
from inspection_system.app.gates.feature_gates import evaluate_feature_gates
from inspection_system.app.inspection_models import (
    GateDecision,
    InspectionOutcome,
    MeasurementBundle,
    RegistrationAssessment,
)
from inspection_system.app.inspection_program import aggregate_lane_results, resolve_inspection_program
from inspection_system.app.lanes import execute_inspection_lane
from inspection_system.app.preprocessing_utils import build_registration_image
from inspection_system.app.registration_engine import register_sample_mask
from inspection_system.app.registration_transform import apply_transform_to_mask, build_transform_summary
from inspection_system.app.reference_service import check_reference_settings_match


logger = logging.getLogger(__name__)


DEFAULT_FEATURE_POSITION_FAMILIES = (*DEFAULT_MOLDED_PART_FEATURE_FAMILIES, "datum_section")


@dataclass
class _PreparedSampleData:
    roi_image: object
    gray: object | None
    sample_mask: object
    roi: tuple[int, int, int, int]
    cv2: object
    np: object


@dataclass
class _ReferenceAssets:
    reference_mask: object
    reference_image: object
    reference_allowed: object
    reference_required: object
    section_masks: list


@dataclass
class _LaneProgramResult:
    lane_results: list[dict]
    lane_aggregation: dict
    primary_lane_result: dict
    active_lane_result: dict
    measurement_result: dict


@dataclass
class _MeasurementBaseline:
    resolved_anomaly_metrics: dict
    metrics: dict
    edge_measurement_mask: object
    edge_measurement_frame: str
    mean_edge_distance_px: float
    section_edge_distances_px: list
    worst_section_edge_distance_px: float
    section_width_ratios: list
    section_center_offsets_px: list
    section_measurement_frame: str
    section_measurements: list[dict]


def _anomaly_evaluation_requested(inspection_cfg: dict) -> bool:
    if not isinstance(inspection_cfg, dict):
        return False

    inspection_mode = str(inspection_cfg.get("inspection_mode", "mask_only")).strip().lower()
    if inspection_mode in {"mask_and_ssim", "mask_and_ml", "full"}:
        return True

    return any(
        inspection_cfg.get(key) is not None
        for key in ("min_ssim", "max_mse", "min_anomaly_score")
    )


def _finite_float(value):
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


ALIGNMENT_PROFILES = {
    "strict": {
        "max_angle_deg": 0.7,
        "max_shift_x": 2,
        "max_shift_y": 2,
    },
    "balanced": {
        "max_angle_deg": 1.0,
        "max_shift_x": 4,
        "max_shift_y": 3,
    },
    "forgiving": {
        "max_angle_deg": 1.8,
        "max_shift_x": 7,
        "max_shift_y": 5,
    },
}


def resolve_alignment_config(config: dict) -> tuple[dict, str]:
    """Resolve alignment config, applying optional tolerance profile defaults."""
    alignment_cfg = dict(config.get("alignment", {}))
    profile_name = str(alignment_cfg.get("tolerance_profile", "balanced")).strip().lower()
    if profile_name not in ALIGNMENT_PROFILES:
        profile_name = "balanced"

    profile_defaults = ALIGNMENT_PROFILES[profile_name]
    for key, value in profile_defaults.items():
        alignment_cfg.setdefault(key, value)

    alignment_cfg["tolerance_profile"] = profile_name
    return alignment_cfg, profile_name


def _compute_edge_mask(mask, np_module):
    white = mask > 0
    if not white.any():
        return white

    padded = np_module.pad(white, 1, mode="constant", constant_values=False)
    interior = white.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            interior &= padded[1 + dy : 1 + dy + white.shape[0], 1 + dx : 1 + dx + white.shape[1]]
    return white & (~interior)


def _mean_nearest_edge_distance(edge_source, edge_target, np_module, cv2_module=None) -> float:
    source_count = int(edge_source.sum())
    target_count = int(edge_target.sum())
    if source_count == 0 and target_count == 0:
        return 0.0
    if source_count == 0 or target_count == 0:
        return float("inf")

    if cv2_module is not None and hasattr(cv2_module, "distanceTransform"):
        distance_input = (~edge_target).astype(np_module.uint8)
        distance_map = cv2_module.distanceTransform(
            distance_input,
            getattr(cv2_module, "DIST_L2", 2),
            getattr(cv2_module, "DIST_MASK_PRECISE", 0),
        )
        return float(distance_map[edge_source].mean())

    source_points = np_module.argwhere(edge_source).astype(np_module.float32)
    target_points = np_module.argwhere(edge_target).astype(np_module.float32)
    deltas = source_points[:, None, :] - target_points[None, :, :]
    distances = np_module.sqrt(np_module.sum(deltas * deltas, axis=2))
    return float(distances.min(axis=1).mean())


def compute_mean_edge_distance_px(reference_mask, sample_mask, np_module, cv2_module=None) -> float:
    reference_edges = _compute_edge_mask(reference_mask, np_module)
    sample_edges = _compute_edge_mask(sample_mask, np_module)
    forward_distance = _mean_nearest_edge_distance(sample_edges, reference_edges, np_module, cv2_module)
    reverse_distance = _mean_nearest_edge_distance(reference_edges, sample_edges, np_module, cv2_module)
    return float((forward_distance + reverse_distance) / 2.0)


def compute_section_edge_distances_px(reference_mask, sample_mask, section_masks, np_module, cv2_module=None):
    distances = []
    for section_mask in section_masks:
        section_white = section_mask > 0
        if not section_white.any():
            continue
        section_reference_mask = np_module.where(section_white, reference_mask, 0).astype(reference_mask.dtype, copy=False)
        section_sample_mask = np_module.where(section_white, sample_mask, 0).astype(sample_mask.dtype, copy=False)
        distances.append(compute_mean_edge_distance_px(section_reference_mask, section_sample_mask, np_module, cv2_module))
    return distances


def compute_section_width_ratios(reference_mask, sample_mask, section_masks, np_module):
    ratios = []
    reference_white = reference_mask > 0
    sample_white = sample_mask > 0
    for section_mask in section_masks:
        section_points = np_module.argwhere(section_mask > 0)
        if section_points.size == 0:
            continue
        y0, x0 = section_points.min(axis=0)
        y1, x1 = section_points.max(axis=0) + 1
        reference_count = int(reference_white[y0:y1, x0:x1].sum())
        if reference_count == 0:
            continue
        sample_count = int(sample_white[y0:y1, x0:x1].sum())
        ratios.append(sample_count / reference_count)
    return ratios


def compute_section_center_offsets_px(reference_mask, sample_mask, section_masks, np_module):
    offsets = []
    reference_white = reference_mask > 0
    sample_white = sample_mask > 0
    for section_mask in section_masks:
        section_white = section_mask > 0
        if not section_white.any():
            continue
        reference_points = np_module.argwhere(reference_white & section_white)
        if reference_points.size == 0:
            continue
        sample_points = np_module.argwhere(sample_white & section_white)
        if sample_points.size == 0:
            offsets.append(float("inf"))
            continue
        reference_center_x = float(reference_points[:, 1].mean())
        sample_center_x = float(sample_points[:, 1].mean())
        offsets.append(abs(sample_center_x - reference_center_x))
    return offsets


def _reference_candidate_rank(passed: bool, details: dict) -> tuple[int, int, float]:
    margins = [
        _finite_float(details.get("required_coverage")) - _finite_float(details.get("effective_min_required_coverage", details.get("min_required_coverage", 0.0))),
        _finite_float(details.get("effective_max_outside_allowed_ratio", details.get("max_outside_allowed_ratio", 0.0))) - _finite_float(details.get("outside_allowed_ratio")),
        _finite_float(details.get("min_section_coverage")) - _finite_float(details.get("effective_min_section_coverage", details.get("min_section_coverage_limit", 0.0))),
    ]

    optional_gates = [
        ("section_center_gate_active", "worst_section_center_offset_px", "effective_max_section_center_offset_px", "max_section_center_offset_px", "lower"),
        ("section_width_gate_active", "worst_section_width_delta_ratio", "effective_max_section_width_delta_ratio", "max_section_width_delta_ratio", "lower"),
        ("section_edge_gate_active", "worst_section_edge_distance_px", "effective_max_section_edge_distance_px", "max_section_edge_distance_px", "lower"),
        ("edge_distance_gate_active", "mean_edge_distance_px", "effective_max_mean_edge_distance_px", "max_mean_edge_distance_px", "lower"),
        ("ssim_gate_active", "ssim", "effective_min_ssim", "min_ssim", "higher"),
        ("mse_gate_active", "mse", "effective_max_mse", "max_mse", "lower"),
        ("anomaly_gate_active", "anomaly_score", "effective_min_anomaly_score", "min_anomaly_score", "higher"),
    ]
    for gate_key, metric_key, effective_key, fallback_key, direction in optional_gates:
        if not details.get(gate_key):
            continue
        value = _finite_float(details.get(metric_key))
        threshold = _finite_float(details.get(effective_key, details.get(fallback_key)))
        if value is None or threshold is None:
            margins.append(-1.0)
            continue
        if direction == "higher":
            margins.append(value - threshold)
        else:
            margins.append(threshold - value)

    if details.get("feature_gate_active"):
        feature_gate_margin = _finite_float(details.get("feature_gate_margin_px"))
        margins.append(feature_gate_margin if feature_gate_margin is not None else -1.0)

    registration = details.get("registration", {}) if isinstance(details.get("registration"), dict) else {}
    if registration.get("rejection_reason"):
        failed_gate_count = max(1, len(registration.get("quality_gate_failures", [])))
        return (1 if passed else 0, -failed_gate_count, -1000.0)

    failed_gate_count = sum(1 for margin in margins if margin < 0)
    return (1 if passed else 0, -failed_gate_count, round(sum(margins), 6))


def _build_reference_candidate_summary(details: dict, passed: bool, rank: tuple[int, int, float]) -> dict:
    registration = details.get("registration", {})
    return {
        "reference_id": details.get("reference_id"),
        "reference_label": details.get("reference_label"),
        "reference_role": details.get("reference_role"),
        "passed": bool(passed),
        "rank": {
            "passed_score": int(rank[0]),
            "failed_gate_score": int(rank[1]),
            "margin_score": float(rank[2]),
        },
        "registration_status": registration.get("status"),
        "registration_runtime_mode": registration.get("runtime_mode"),
        "registration_applied_strategy": registration.get("applied_strategy"),
        "registration_datum_frame": registration.get("datum_frame"),
        "registration_rejected": bool(registration.get("rejection_reason")),
        "registration_rejection_reason": registration.get("rejection_reason"),
        "edge_measurement_frame": details.get("edge_measurement_frame"),
        "section_measurement_frame": details.get("section_measurement_frame"),
        "required_coverage": float(details.get("required_coverage", 0.0)),
        "outside_allowed_ratio": float(details.get("outside_allowed_ratio", 0.0)),
        "min_section_coverage": float(details.get("min_section_coverage", 0.0)),
        "mean_edge_distance_px": details.get("mean_edge_distance_px"),
        "worst_section_edge_distance_px": details.get("worst_section_edge_distance_px"),
        "worst_section_width_delta_ratio": details.get("worst_section_width_delta_ratio"),
        "worst_section_center_offset_px": details.get("worst_section_center_offset_px"),
    }


def _find_lane_result(lane_results: list[dict], lane_id: str | None) -> dict:
    if lane_id is not None:
        for lane_result in lane_results:
            if str(lane_result.get("lane_id")) == str(lane_id):
                return lane_result
    return lane_results[0]


def _serialize_lane_result(lane_result: dict) -> dict:
    measurement_result = lane_result.get("measurement_result", {})
    return {
        "lane_id": lane_result.get("lane_id"),
        "lane_type": lane_result.get("lane_type"),
        "authoritative": bool(lane_result.get("authoritative", False)),
        "passed": bool(lane_result.get("passed", False)),
        "inspection_failure_cause": lane_result.get("inspection_failure_cause"),
        "threshold_summary": deepcopy(lane_result.get("threshold_summary", {})),
        "feature_measurements": deepcopy(lane_result.get("feature_measurements", [])),
        "feature_position_summary": deepcopy(lane_result.get("feature_position_summary")),
        "edge_measurement_frame": lane_result.get("edge_measurement_frame"),
        "section_measurement_frame": lane_result.get("section_measurement_frame"),
        "mean_edge_distance_px": measurement_result.get("mean_edge_distance_px"),
        "worst_section_edge_distance_px": measurement_result.get("worst_section_edge_distance_px"),
        "worst_section_width_delta_ratio": measurement_result.get("worst_section_width_delta_ratio"),
        "worst_section_center_offset_px": measurement_result.get("worst_section_center_offset_px"),
    }


def _build_feature_position_measurements(section_measurements: list[dict], section_measurement_frame: str) -> tuple[list[dict], dict | None]:
    if section_measurement_frame != "datum":
        return [], None

    feature_measurements: list[dict] = []
    for index, measurement in enumerate(section_measurements):
        if not isinstance(measurement, dict):
            continue

        section_index = int(measurement.get("section_index", index))
        sample_detected = bool(measurement.get("sample_detected", False))
        center_offset_px = _finite_float(measurement.get("center_offset_px")) if sample_detected else None
        feature_measurements.append(
            {
                "feature_key": f"section_{section_index + 1}",
                "feature_label": f"Datum Section {section_index + 1}",
                "feature_family": "datum_section",
                "feature_type": "datum_section_position",
                "section_index": section_index,
                "measurement_frame": section_measurement_frame,
                "sample_detected": sample_detected,
                "failure_cause": "feature_not_found" if not sample_detected else "feature_position",
                "reference_center": measurement.get("reference_center"),
                "expected_sample_window": measurement.get("expected_sample_window"),
                "observed_center_reference": measurement.get("observed_center_reference"),
                "center_offset_px": center_offset_px,
            }
        )

    if not feature_measurements:
        return [], None

    def _rank(entry: dict) -> tuple[int, float]:
        if not entry.get("sample_detected", False):
            return (1, float("inf"))
        return (0, float(entry.get("center_offset_px") or 0.0))

    worst_feature = max(feature_measurements, key=_rank)
    return feature_measurements, {
        "feature_label": worst_feature["feature_label"],
        "feature_family": worst_feature["feature_family"],
        "feature_type": "datum_section_position",
        "measurement_frame": section_measurement_frame,
        "feature_count": len(feature_measurements),
        "feature_key": worst_feature["feature_key"],
        "section_index": int(worst_feature["section_index"]),
        "sample_detected": bool(worst_feature["sample_detected"]),
        "failure_cause": str(worst_feature["failure_cause"]),
        "reference_center": worst_feature.get("reference_center"),
        "expected_sample_window": worst_feature.get("expected_sample_window"),
        "observed_center_reference": worst_feature.get("observed_center_reference"),
        "center_offset_px": worst_feature.get("center_offset_px"),
    }


def _resolve_feature_position_families(inspection_cfg: dict) -> list[str]:
    raw_value = inspection_cfg.get("feature_position_families", DEFAULT_FEATURE_POSITION_FAMILIES)
    if isinstance(raw_value, str):
        candidates = [segment.strip() for segment in raw_value.split(",")]
    elif isinstance(raw_value, (list, tuple)):
        candidates = [str(segment).strip() for segment in raw_value]
    else:
        candidates = list(DEFAULT_FEATURE_POSITION_FAMILIES)

    resolved: list[str] = []
    valid_names = set(DEFAULT_FEATURE_POSITION_FAMILIES)
    for candidate in candidates:
        normalized = str(candidate or "").strip().lower()
        if normalized in valid_names and normalized not in resolved:
            resolved.append(normalized)

    if not resolved:
        return list(DEFAULT_FEATURE_POSITION_FAMILIES)
    return resolved


def _resolve_inspection_failure_cause(
    registration_failed: bool,
    threshold_summary: dict,
    feature_position_summary: dict | None,
) -> str | None:
    if registration_failed:
        return "registration_failure"

    if bool(threshold_summary.get("feature_gate_active", False)) and not bool(
        threshold_summary.get("feature_gate_passed", True)
    ):
        if feature_position_summary is not None:
            return str(
                feature_position_summary.get(
                    "failure_cause",
                    "feature_not_found" if not feature_position_summary.get("sample_detected", True) else "feature_position",
                )
            )
        return str(threshold_summary.get("feature_gate_failure_cause") or "feature_position")

    if bool(threshold_summary.get("section_center_gate_active", False)):
        observed = _finite_float(threshold_summary.get("worst_section_center_offset_px"))
        allowed = _finite_float(threshold_summary.get("max_section_center_offset_px"))
        if observed is None or allowed is None or observed > allowed:
            if feature_position_summary is not None:
                return str(
                    feature_position_summary.get(
                        "failure_cause",
                        "feature_not_found" if not feature_position_summary.get("sample_detected", True) else "feature_position",
                    )
                )
            return "feature_position"

    return None


def _measure_inspection_outcome(
    sample_mask,
    aligned_sample_mask,
    transform_summary: dict | None,
    reference_mask,
    reference_allowed,
    reference_required,
    section_masks,
    inspection_cfg: dict,
    score_sample,
    evaluate_metrics,
    anomaly_metrics: dict | None,
    cv2,
    np,
    baseline_measurements: _MeasurementBaseline | None = None,
) -> dict:
    if baseline_measurements is None:
        baseline_measurements = _compute_measurement_baseline(
            sample_mask,
            aligned_sample_mask,
            transform_summary,
            reference_mask,
            reference_allowed,
            reference_required,
            section_masks,
            score_sample,
            anomaly_metrics,
            cv2,
            np,
        )

    resolved_anomaly_metrics = dict(baseline_measurements.resolved_anomaly_metrics)
    metrics = baseline_measurements.metrics
    edge_measurement_mask = baseline_measurements.edge_measurement_mask
    edge_measurement_frame = baseline_measurements.edge_measurement_frame
    mean_edge_distance_px = baseline_measurements.mean_edge_distance_px
    section_edge_distances_px = baseline_measurements.section_edge_distances_px
    worst_section_edge_distance_px = baseline_measurements.worst_section_edge_distance_px
    section_width_ratios = baseline_measurements.section_width_ratios
    section_center_offsets_px = baseline_measurements.section_center_offsets_px
    section_measurement_frame = baseline_measurements.section_measurement_frame
    section_measurements = baseline_measurements.section_measurements

    feature_measurements: list[dict] = []
    feature_position_summary = None
    feature_position_families = _resolve_feature_position_families(inspection_cfg)
    molded_part_feature_families = [
        family_name for family_name in feature_position_families if family_name != "datum_section"
    ]
    if section_measurement_frame == "datum" and molded_part_feature_families:
        feature_measurements, feature_position_summary = extract_molded_part_feature_measurements(
            reference_required,
            edge_measurement_mask,
            molded_part_feature_families,
            cv2,
            np,
        )
    if not feature_measurements and "datum_section" in feature_position_families:
        feature_measurements, feature_position_summary = _build_feature_position_measurements(
            section_measurements,
            section_measurement_frame,
        )

    worst_section_width_delta_ratio = max((abs(float(ratio) - 1.0) for ratio in section_width_ratios), default=0.0)
    worst_section_center_offset_px = max(section_center_offsets_px, default=0.0)
    metric_inputs = {
        **metrics,
        "mean_edge_distance_px": mean_edge_distance_px,
        "worst_section_edge_distance_px": worst_section_edge_distance_px,
        "worst_section_width_delta_ratio": worst_section_width_delta_ratio,
        "worst_section_center_offset_px": worst_section_center_offset_px,
        **resolved_anomaly_metrics,
    }
    feature_gate_result = evaluate_feature_gates(feature_measurements, inspection_cfg)
    if feature_gate_result["feature_position_summary"] is not None:
        feature_position_summary = feature_gate_result["feature_position_summary"]
    passed, threshold_summary = evaluate_metrics(metric_inputs, inspection_cfg)
    threshold_summary = {
        **threshold_summary,
        **feature_gate_result["summary"],
    }
    passed = bool(passed and feature_gate_result["passed"])
    return {
        "passed": bool(passed),
        "threshold_summary": threshold_summary,
        "metrics": metrics,
        "mean_edge_distance_px": mean_edge_distance_px,
        "section_edge_distances_px": section_edge_distances_px,
        "worst_section_edge_distance_px": worst_section_edge_distance_px,
        "section_width_ratios": section_width_ratios,
        "section_center_offsets_px": section_center_offsets_px,
        "section_measurement_frame": section_measurement_frame,
        "section_measurements": section_measurements,
        "feature_measurements": feature_measurements,
        "feature_position_summary": feature_position_summary,
        "worst_section_width_delta_ratio": worst_section_width_delta_ratio,
        "worst_section_center_offset_px": worst_section_center_offset_px,
        "edge_measurement_frame": edge_measurement_frame,
    }


def _compute_measurement_baseline(
    sample_mask,
    aligned_sample_mask,
    transform_summary: dict | None,
    reference_mask,
    reference_allowed,
    reference_required,
    section_masks,
    score_sample,
    anomaly_metrics: dict | None,
    cv2,
    np,
) -> _MeasurementBaseline:
    resolved_anomaly_metrics = dict(anomaly_metrics or {})
    metrics = score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks)
    datum_section_metrics = None
    edge_measurement_mask = aligned_sample_mask
    edge_measurement_frame = "aligned_mask"
    if transform_summary is not None:
        datum_section_metrics = compute_datum_section_measurements(
            sample_mask,
            section_masks,
            transform_summary,
            np,
        )
        edge_measurement_mask = apply_transform_to_mask(
            sample_mask,
            transform_summary,
            cv2,
            np,
        )
        edge_measurement_frame = "datum"

    mean_edge_distance_px = compute_mean_edge_distance_px(reference_mask, edge_measurement_mask, np, cv2)
    section_edge_distances_px = compute_section_edge_distances_px(
        reference_mask,
        edge_measurement_mask,
        section_masks,
        np,
        cv2,
    )
    worst_section_edge_distance_px = max(section_edge_distances_px) if section_edge_distances_px else 0.0

    if datum_section_metrics is not None:
        section_width_ratios = datum_section_metrics["section_width_ratios"]
        section_center_offsets_px = datum_section_metrics["section_center_offsets_px"]
        section_measurement_frame = datum_section_metrics["frame"]
        section_measurements = datum_section_metrics["section_measurements"]
    else:
        section_width_ratios = compute_section_width_ratios(reference_mask, aligned_sample_mask, section_masks, np)
        section_center_offsets_px = compute_section_center_offsets_px(reference_mask, aligned_sample_mask, section_masks, np)
        section_measurement_frame = "aligned_mask"
        section_measurements = []

    return _MeasurementBaseline(
        resolved_anomaly_metrics=resolved_anomaly_metrics,
        metrics=metrics,
        edge_measurement_mask=edge_measurement_mask,
        edge_measurement_frame=edge_measurement_frame,
        mean_edge_distance_px=mean_edge_distance_px,
        section_edge_distances_px=section_edge_distances_px,
        worst_section_edge_distance_px=worst_section_edge_distance_px,
        section_width_ratios=section_width_ratios,
        section_center_offsets_px=section_center_offsets_px,
        section_measurement_frame=section_measurement_frame,
        section_measurements=section_measurements,
    )


def _should_prefer_coarse_moments_measurement(
    refined_measurement: dict,
    coarse_measurement: dict,
    refined_transform: dict,
    coarse_transform: dict,
) -> bool:
    if not refined_measurement.get("passed") or coarse_measurement.get("passed"):
        return False

    coarse_summary = coarse_measurement.get("threshold_summary", {})
    refined_summary = refined_measurement.get("threshold_summary", {})
    coarse_width = coarse_summary.get("worst_section_width_delta_ratio")
    coarse_width_limit = coarse_summary.get("effective_max_section_width_delta_ratio")
    if coarse_width is None or coarse_width_limit in (None, ""):
        return False

    width_excess = float(coarse_width) - float(coarse_width_limit)
    if width_excess < 0.5:
        return False

    transform_delta_shift = math.hypot(
        float(refined_transform.get("shift_x", 0)) - float(coarse_transform.get("shift_x", 0)),
        float(refined_transform.get("shift_y", 0)) - float(coarse_transform.get("shift_y", 0)),
    )
    transform_delta_angle = abs(float(refined_transform.get("angle_deg", 0.0)) - float(coarse_transform.get("angle_deg", 0.0)))
    if transform_delta_shift < 0.5 and transform_delta_angle < 0.15:
        return False

    outside_improvement = float(coarse_summary.get("outside_allowed_ratio", 0.0)) - float(
        refined_summary.get("outside_allowed_ratio", 0.0)
    )
    edge_improvement = float(coarse_summary.get("mean_edge_distance_px", 0.0)) - float(
        refined_summary.get("mean_edge_distance_px", 0.0)
    )
    coverage_improvement = float(refined_summary.get("required_coverage", 0.0)) - float(
        coarse_summary.get("required_coverage", 0.0)
    )

    if outside_improvement >= 0.004:
        return False
    if edge_improvement >= 0.75:
        return False
    if coverage_improvement >= 0.01:
        return False
    return True


def _check_reference_settings_warning(config: dict) -> None:
    try:
        settings_match, mismatch_msg = check_reference_settings_match(config)
        if not settings_match:
            logger.warning(f"Reference settings mismatch: {mismatch_msg}")
    except Exception as exc:
        logger.debug(f"Could not check reference settings: {exc}")


def _prepare_sample_data(
    image_path: Path,
    inspection_cfg: dict,
    make_binary_mask,
    import_cv2_and_numpy,
    dilate_mask,
    erode_mask,
) -> _PreparedSampleData:
    roi_image, gray, sample_mask, roi, cv2, np = make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy)

    sample_erode_iterations = int(inspection_cfg.get("sample_erode_iterations", 1))
    sample_dilate_iterations = int(inspection_cfg.get("sample_dilate_iterations", 1))
    sample_mask = erode_mask(sample_mask, sample_erode_iterations, cv2, np)
    sample_mask = dilate_mask(sample_mask, sample_dilate_iterations, cv2, np)

    return _PreparedSampleData(
        roi_image=roi_image,
        gray=gray,
        sample_mask=sample_mask,
        roi=roi,
        cv2=cv2,
        np=np,
    )


def _load_reference_images(
    reference_mask_path: Path,
    reference_image_path: Path,
    sample_mask,
    cv2,
) -> tuple[object, object]:
    reference_mask = cv2.imread(str(reference_mask_path), cv2.IMREAD_GRAYSCALE)
    if reference_mask is None:
        raise FileNotFoundError(f"Reference mask not found: {reference_mask_path}")

    reference_image = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
    if reference_image is None:
        raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

    if reference_mask.shape != sample_mask.shape:
        raise ValueError(
            f"Reference mask shape {reference_mask.shape} does not match sample mask shape {sample_mask.shape}."
        )

    return reference_mask, reference_image


def _build_reference_assets(
    reference_mask,
    inspection_cfg: dict,
    build_reference_regions,
    compute_section_masks,
    dilate_mask,
    erode_mask,
    cv2,
    np,
) -> _ReferenceAssets:
    reference_allowed, reference_required = build_reference_regions(
        reference_mask,
        inspection_cfg,
        lambda mask, iterations: dilate_mask(mask, iterations, cv2, np),
        lambda mask, iterations: erode_mask(mask, iterations, cv2, np),
    )

    section_masks = compute_section_masks(
        reference_required,
        int(inspection_cfg.get("section_columns", 12)),
        cv2,
        np,
    )

    return _ReferenceAssets(
        reference_mask=reference_mask,
        reference_image=None,
        reference_allowed=reference_allowed,
        reference_required=reference_required,
        section_masks=section_masks,
    )


def _build_anomaly_metrics_provider(
    inspection_cfg: dict,
    registration_result,
    roi_image,
    reference_image,
    sample_mask,
    anomaly_detector,
):
    anomaly_metrics_cache: dict | None = None

    def _get_anomaly_metrics() -> dict:
        nonlocal anomaly_metrics_cache

        if anomaly_metrics_cache is not None:
            return anomaly_metrics_cache

        if registration_result.rejection_reason or not _anomaly_evaluation_requested(inspection_cfg):
            anomaly_metrics_cache = {}
            return anomaly_metrics_cache

        anomaly_metrics_cache = detect_anomalies(roi_image, reference_image, sample_mask, anomaly_detector)
        return anomaly_metrics_cache

    return _get_anomaly_metrics


def _execute_measurement_program(
    inspection_program,
    inspection_cfg: dict,
    sample_mask,
    aligned_sample_mask,
    transform_summary,
    reference_assets: _ReferenceAssets,
    score_sample,
    evaluate_metrics,
    anomaly_metrics_provider,
    cv2,
    np,
) -> _LaneProgramResult:
    baseline_measurements = _compute_measurement_baseline(
        sample_mask,
        aligned_sample_mask,
        transform_summary,
        reference_assets.reference_mask,
        reference_assets.reference_allowed,
        reference_assets.reference_required,
        reference_assets.section_masks,
        score_sample,
        anomaly_metrics_provider(),
        cv2,
        np,
    )

    lane_results: list[dict] = []
    for lane in inspection_program.lanes:
        lane_result = execute_inspection_lane(
            lane,
            base_inspection_cfg=inspection_cfg,
            measure_lane=lambda lane_inspection_cfg: _measure_inspection_outcome(
                sample_mask,
                aligned_sample_mask,
                transform_summary,
                reference_assets.reference_mask,
                reference_assets.reference_allowed,
                reference_assets.reference_required,
                reference_assets.section_masks,
                lane_inspection_cfg,
                score_sample,
                evaluate_metrics,
                baseline_measurements.resolved_anomaly_metrics,
                cv2,
                np,
                baseline_measurements=baseline_measurements,
            ),
        )
        lane_result["inspection_failure_cause"] = _resolve_inspection_failure_cause(
            False,
            lane_result["threshold_summary"],
            lane_result["feature_position_summary"],
        )
        lane_results.append(lane_result)

    lane_aggregation = aggregate_lane_results(inspection_program, lane_results)
    primary_lane_result = _find_lane_result(lane_results, lane_aggregation["primary_lane_id"])
    active_lane_result = _find_lane_result(lane_results, lane_aggregation["active_lane_id"])
    return _LaneProgramResult(
        lane_results=lane_results,
        lane_aggregation=lane_aggregation,
        primary_lane_result=primary_lane_result,
        active_lane_result=active_lane_result,
        measurement_result=active_lane_result["measurement_result"],
    )


def _build_debug_diff(aligned_sample_mask, metrics: dict, reference_allowed, reference_required, np):
    required_white = reference_required > 0
    allowed_white = reference_allowed > 0

    image_size = aligned_sample_mask.shape[0] * aligned_sample_mask.shape[1] * 3
    max_reasonable_pixels = 50 * 1024 * 1024
    if image_size > max_reasonable_pixels:
        raise MemoryError(f"Image too large for processing: {image_size} pixels exceeds {max_reasonable_pixels} limit")

    diff = np.zeros((aligned_sample_mask.shape[0], aligned_sample_mask.shape[1], 3), dtype=np.uint8)
    diff[allowed_white] = (0, 80, 0)
    diff[required_white] = (0, 255, 0)
    diff[metrics["missing_required_mask"]] = (0, 0, 255)
    diff[metrics["outside_allowed_mask"]] = (255, 0, 0)
    return diff


def _maybe_save_debug_outputs(
    image_path: Path,
    inspection_cfg: dict,
    aligned_sample_mask,
    metrics: dict,
    reference_assets: _ReferenceAssets,
    save_debug_outputs,
    np,
) -> dict:
    if not bool(inspection_cfg.get("save_debug_images", True)):
        return {}

    diff = _build_debug_diff(
        aligned_sample_mask,
        metrics,
        reference_assets.reference_allowed,
        reference_assets.reference_required,
        np,
    )
    return save_debug_outputs(image_path.stem, aligned_sample_mask, diff)


def inspect_against_reference(
    config: dict,
    image_path: Path,
    make_binary_mask,
    reference_mask_path: Path,
    reference_image_path: Path,
    align_sample_mask,
    build_reference_regions,
    compute_section_masks,
    score_sample,
    evaluate_metrics,
    save_debug_outputs,
    import_cv2_and_numpy,
    dilate_mask,
    erode_mask,
    anomaly_detector=None,
    *,
    prepared_sample_data: _PreparedSampleData | None = None,
    reference_settings_checked: bool = False,
) -> tuple[bool, dict]:
    inspection_cfg = config.get("inspection", {})
    alignment_cfg, alignment_profile = resolve_alignment_config(config)
    if not reference_settings_checked:
        _check_reference_settings_warning(config)

    sample_data = prepared_sample_data or _prepare_sample_data(
        image_path,
        inspection_cfg,
        make_binary_mask,
        import_cv2_and_numpy,
        dilate_mask,
        erode_mask,
    )
    roi_image = sample_data.roi_image
    gray = sample_data.gray
    sample_mask = sample_data.sample_mask
    roi = sample_data.roi
    cv2 = sample_data.cv2
    np = sample_data.np

    reference_mask, reference_image = _load_reference_images(
        reference_mask_path,
        reference_image_path,
        sample_mask,
        cv2,
    )

    sample_registration_image = build_registration_image(gray if gray is not None else roi_image, sample_mask, np)
    reference_registration_image = build_registration_image(reference_image, reference_mask, np)

    registration_result = register_sample_mask(
        sample_mask,
        reference_mask,
        alignment_cfg,
        cv2,
        np,
        align_sample_mask,
        sample_registration_image=sample_registration_image,
        reference_registration_image=reference_registration_image,
    )
    aligned_sample_mask = registration_result.aligned_mask
    best_angle_deg = registration_result.angle_deg
    best_shift_x = registration_result.shift_x
    best_shift_y = registration_result.shift_y
    effective_transform_summary = registration_result.transform if registration_result.status == "aligned" else None
    registration_guard_reason = None

    reference_assets = _build_reference_assets(
        reference_mask,
        inspection_cfg,
        build_reference_regions,
        compute_section_masks,
        dilate_mask,
        erode_mask,
        cv2,
        np,
    )
    reference_assets.reference_image = reference_image

    inspection_program = resolve_inspection_program(config)
    anomaly_metrics_provider = _build_anomaly_metrics_provider(
        inspection_cfg,
        registration_result,
        roi_image,
        reference_image,
        sample_mask,
        anomaly_detector,
    )

    lane_program_result = _execute_measurement_program(
        inspection_program,
        inspection_cfg,
        sample_mask,
        aligned_sample_mask,
        effective_transform_summary,
        reference_assets,
        score_sample,
        evaluate_metrics,
        anomaly_metrics_provider,
        cv2,
        np,
    )
    lane_results = lane_program_result.lane_results
    lane_aggregation = lane_program_result.lane_aggregation
    primary_lane_result = lane_program_result.primary_lane_result
    active_lane_result = lane_program_result.active_lane_result
    measurement_result = lane_program_result.measurement_result

    if registration_result.status == "aligned" and registration_result.runtime_mode == "moments":
        coarse_aligned_sample_mask, coarse_angle_deg, coarse_shift_x, coarse_shift_y = align_sample_mask(
            sample_mask,
            reference_mask,
            alignment_cfg,
            cv2,
            np,
        )
        coarse_transform_summary = build_transform_summary(
            sample_mask.shape[:2],
            coarse_angle_deg,
            coarse_shift_x,
            coarse_shift_y,
        )
        if (
            abs(float(coarse_angle_deg) - float(best_angle_deg)) > 1e-6
            or int(coarse_shift_x) != int(best_shift_x)
            or int(coarse_shift_y) != int(best_shift_y)
        ):
            coarse_lane_program_result = _execute_measurement_program(
                inspection_program,
                inspection_cfg,
                sample_mask,
                coarse_aligned_sample_mask,
                coarse_transform_summary,
                reference_assets,
                score_sample,
                evaluate_metrics,
                anomaly_metrics_provider,
                cv2,
                np,
            )
            coarse_primary_lane_result = coarse_lane_program_result.primary_lane_result
            coarse_measurement_result = coarse_lane_program_result.measurement_result
            if _should_prefer_coarse_moments_measurement(
                primary_lane_result["measurement_result"],
                coarse_measurement_result,
                registration_result.transform,
                coarse_transform_summary,
            ):
                registration_guard_reason = (
                    "Moments refinement was ignored for scoring because it only removed a large section-width drift "
                    "without enough supporting improvement in independent metrics."
                )
                lane_results = coarse_lane_program_result.lane_results
                lane_aggregation = coarse_lane_program_result.lane_aggregation
                primary_lane_result = coarse_lane_program_result.primary_lane_result
                active_lane_result = coarse_lane_program_result.active_lane_result
                measurement_result = coarse_lane_program_result.measurement_result
                aligned_sample_mask = coarse_aligned_sample_mask
                best_angle_deg = float(coarse_angle_deg)
                best_shift_x = int(coarse_shift_x)
                best_shift_y = int(coarse_shift_y)
                effective_transform_summary = coarse_transform_summary

    metrics = measurement_result["metrics"]
    mean_edge_distance_px = measurement_result["mean_edge_distance_px"]
    section_edge_distances_px = measurement_result["section_edge_distances_px"]
    worst_section_edge_distance_px = measurement_result["worst_section_edge_distance_px"]
    section_width_ratios = measurement_result["section_width_ratios"]
    section_center_offsets_px = measurement_result["section_center_offsets_px"]
    section_measurement_frame = measurement_result["section_measurement_frame"]
    section_measurements = measurement_result["section_measurements"]
    worst_section_width_delta_ratio = measurement_result["worst_section_width_delta_ratio"]
    worst_section_center_offset_px = measurement_result["worst_section_center_offset_px"]
    edge_measurement_frame = measurement_result["edge_measurement_frame"]
    passed = bool(lane_aggregation["passed"])
    threshold_summary = measurement_result["threshold_summary"]
    measurement_bundle = MeasurementBundle.from_legacy(measurement_result)
    gate_decision = GateDecision.from_legacy(passed, threshold_summary)
    registration_failed = bool(registration_result.rejection_reason)
    if registration_failed:
        passed = False

    inspection_failure_cause = _resolve_inspection_failure_cause(
        registration_failed,
        threshold_summary,
        measurement_result["feature_position_summary"],
    )

    required_coverage = float(threshold_summary["required_coverage"])
    outside_allowed_ratio = float(threshold_summary["outside_allowed_ratio"])
    min_section_coverage = float(threshold_summary["min_section_coverage"])

    min_required_coverage = float(threshold_summary["min_required_coverage"])
    max_outside_allowed_ratio = float(threshold_summary["max_outside_allowed_ratio"])
    min_section_coverage_limit = float(threshold_summary["min_section_coverage_limit"])
    debug_paths = _maybe_save_debug_outputs(
        image_path,
        inspection_cfg,
        aligned_sample_mask,
        metrics,
        reference_assets,
        save_debug_outputs,
        np,
    )

    registration_assessment = RegistrationAssessment.from_registration_result(
        registration_result,
        transform=effective_transform_summary or registration_result.transform,
        scoring_guard_reason=registration_guard_reason,
    )

    details = {
        "roi": {
            "x": roi[0],
            "y": roi[1],
            "width": roi[2],
            "height": roi[3],
        },
        "best_angle_deg": best_angle_deg,
        "best_shift_x": best_shift_x,
        "best_shift_y": best_shift_y,
        "alignment_profile": alignment_profile,
        "registration": registration_assessment.to_legacy_dict(),
        "failure_stage": "registration" if registration_failed else "inspection",
        "inspection_failure_cause": inspection_failure_cause,
        "inspection_program": {
            "program_id": inspection_program.program_id,
            "aggregation_policy": inspection_program.aggregation_policy,
            "lane_ids": [lane.lane_id for lane in inspection_program.lanes],
            "primary_lane_id": lane_aggregation["primary_lane_id"],
            "active_lane_id": lane_aggregation["active_lane_id"],
        },
        "lane_results": [_serialize_lane_result(lane_result) for lane_result in lane_results],
        "failed_lane_ids": list(lane_aggregation["failed_lane_ids"]),
        "failed_authoritative_lane_ids": list(lane_aggregation["failed_authoritative_lane_ids"]),
        "failed_advisory_lane_ids": list(lane_aggregation["failed_advisory_lane_ids"]),
        "required_coverage": required_coverage,
        "outside_allowed_ratio": outside_allowed_ratio,
        "min_section_coverage": min_section_coverage,
        "section_coverages": metrics["section_coverages"],
        "sample_white_pixels": metrics["sample_white_pixels"],
        "min_required_coverage": min_required_coverage,
        "max_outside_allowed_ratio": max_outside_allowed_ratio,
        "min_section_coverage_limit": min_section_coverage_limit,
        "effective_min_required_coverage": threshold_summary.get("effective_min_required_coverage", min_required_coverage),
        "effective_max_outside_allowed_ratio": threshold_summary.get("effective_max_outside_allowed_ratio", max_outside_allowed_ratio),
        "effective_min_section_coverage": threshold_summary.get("effective_min_section_coverage", min_section_coverage_limit),
        "mean_edge_distance_px": threshold_summary.get("mean_edge_distance_px", mean_edge_distance_px),
        "edge_measurement_frame": edge_measurement_frame,
        "max_mean_edge_distance_px": threshold_summary.get("max_mean_edge_distance_px"),
        "effective_max_mean_edge_distance_px": threshold_summary.get("effective_max_mean_edge_distance_px"),
        "section_edge_distances_px": section_edge_distances_px,
        "worst_section_edge_distance_px": threshold_summary.get(
            "worst_section_edge_distance_px",
            worst_section_edge_distance_px,
        ),
        "max_section_edge_distance_px": threshold_summary.get("max_section_edge_distance_px"),
        "effective_max_section_edge_distance_px": threshold_summary.get("effective_max_section_edge_distance_px"),
        "section_width_ratios": section_width_ratios,
        "section_measurement_frame": section_measurement_frame,
        "section_measurements": section_measurements,
        "worst_section_width_delta_ratio": threshold_summary.get(
            "worst_section_width_delta_ratio",
            worst_section_width_delta_ratio,
        ),
        "max_section_width_delta_ratio": threshold_summary.get("max_section_width_delta_ratio"),
        "effective_max_section_width_delta_ratio": threshold_summary.get("effective_max_section_width_delta_ratio"),
        "section_center_offsets_px": section_center_offsets_px,
        "feature_measurements": measurement_bundle.feature_measurements,
        "feature_position_summary": measurement_bundle.feature_position_summary,
        "worst_section_center_offset_px": threshold_summary.get(
            "worst_section_center_offset_px",
            worst_section_center_offset_px,
        ),
        "max_section_center_offset_px": threshold_summary.get("max_section_center_offset_px"),
        "effective_max_section_center_offset_px": threshold_summary.get("effective_max_section_center_offset_px"),
        "min_ssim": threshold_summary.get("min_ssim"),
        "max_mse": threshold_summary.get("max_mse"),
        "min_anomaly_score": threshold_summary.get("min_anomaly_score"),
        "effective_min_ssim": threshold_summary.get("effective_min_ssim"),
        "effective_max_mse": threshold_summary.get("effective_max_mse"),
        "effective_min_anomaly_score": threshold_summary.get("effective_min_anomaly_score"),
        "inspection_mode": threshold_summary.get("inspection_mode", "mask_only"),
        "blend_mode": threshold_summary.get("blend_mode", "hard_only"),
        "tolerance_mode": threshold_summary.get("tolerance_mode", "balanced"),
        "learned_ranges_active": bool(threshold_summary.get("learned_ranges_active", False)),
        "edge_distance_gate_active": bool(threshold_summary.get("edge_distance_gate_active", False)),
        "section_edge_gate_active": bool(threshold_summary.get("section_edge_gate_active", False)),
        "section_width_gate_active": bool(threshold_summary.get("section_width_gate_active", False)),
        "section_center_gate_active": bool(threshold_summary.get("section_center_gate_active", False)),
        "ssim_gate_active": bool(threshold_summary.get("ssim_gate_active", False)),
        "mse_gate_active": bool(threshold_summary.get("mse_gate_active", False)),
        "anomaly_gate_active": bool(threshold_summary.get("anomaly_gate_active", False)),
        "feature_gate_active": bool(threshold_summary.get("feature_gate_active", False)),
        "feature_gate_passed": bool(threshold_summary.get("feature_gate_passed", True)),
        "feature_gate_failed_check_count": int(threshold_summary.get("feature_gate_failed_check_count", 0)),
        "feature_gate_metric": threshold_summary.get("feature_gate_metric"),
        "feature_gate_feature_key": threshold_summary.get("feature_gate_feature_key"),
        "feature_gate_failure_cause": threshold_summary.get("feature_gate_failure_cause"),
        "feature_gate_observed_value": threshold_summary.get("feature_gate_observed_value"),
        "feature_gate_threshold": threshold_summary.get("feature_gate_threshold"),
        "feature_gate_margin_px": threshold_summary.get("feature_gate_margin_px"),
        "feature_gate_ratio": threshold_summary.get("feature_gate_ratio"),
        "max_feature_dx_px": threshold_summary.get("max_feature_dx_px"),
        "max_feature_dy_px": threshold_summary.get("max_feature_dy_px"),
        "max_feature_radial_offset_px": threshold_summary.get("max_feature_radial_offset_px"),
        "max_feature_pair_spacing_delta_px": threshold_summary.get("max_feature_pair_spacing_delta_px"),
        "debug_paths": debug_paths,
        **anomaly_metrics_provider(),
    }
    inspection_outcome = InspectionOutcome.from_legacy_details(
        passed=passed,
        registration=registration_assessment,
        measurements=measurement_bundle,
        gate_decision=gate_decision,
        details=details,
    )
    return inspection_outcome.passed, inspection_outcome.to_legacy_details()


def inspect_against_references(
    config: dict,
    image_path: Path,
    reference_candidates: list[dict],
    make_binary_mask,
    align_sample_mask,
    build_reference_regions,
    compute_section_masks,
    score_sample,
    evaluate_metrics,
    save_debug_outputs,
    import_cv2_and_numpy,
    dilate_mask,
    erode_mask,
    anomaly_detector=None,
) -> tuple[bool, dict]:
    if not reference_candidates:
        raise FileNotFoundError("No reference candidates are available for inspection.")

    prepared_sample_data = None
    reference_settings_checked = False
    if all(callable(func) for func in (make_binary_mask, import_cv2_and_numpy, dilate_mask, erode_mask)):
        inspection_cfg = config.get("inspection", {})
        _check_reference_settings_warning(config)
        reference_settings_checked = True
        prepared_sample_data = _prepare_sample_data(
            image_path,
            inspection_cfg,
            make_binary_mask,
            import_cv2_and_numpy,
            dilate_mask,
            erode_mask,
        )

    ranked_results: list[tuple[tuple[int, int, float], bool, dict]] = []
    candidate_summaries: list[dict] = []
    errors: list[str] = []
    evaluated_ids: list[str] = []

    for candidate in reference_candidates:
        reference_id = str(candidate.get("reference_id", "unknown"))
        evaluated_ids.append(reference_id)
        try:
            passed, details = inspect_against_reference(
                config,
                image_path,
                make_binary_mask,
                Path(candidate["reference_mask_path"]),
                Path(candidate["reference_image_path"]),
                align_sample_mask,
                build_reference_regions,
                compute_section_masks,
                score_sample,
                evaluate_metrics,
                save_debug_outputs,
                import_cv2_and_numpy,
                dilate_mask,
                erode_mask,
                anomaly_detector=anomaly_detector,
                prepared_sample_data=prepared_sample_data,
                reference_settings_checked=reference_settings_checked,
            )
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{reference_id}: {exc}")
            continue

        details["reference_id"] = reference_id
        details["reference_label"] = str(candidate.get("label", reference_id))
        details["reference_role"] = str(candidate.get("role", "candidate"))
        details["reference_mask_path"] = str(candidate.get("reference_mask_path"))
        details["reference_image_path"] = str(candidate.get("reference_image_path"))
        details["reference_candidate_count"] = len(reference_candidates)
        details["evaluated_reference_ids"] = list(evaluated_ids)
        details["reference_strategy"] = str(config.get("inspection", {}).get("reference_strategy", "golden_only"))
        rank = _reference_candidate_rank(passed, details)
        ranked_results.append((rank, passed, details))
        candidate_summaries.append(_build_reference_candidate_summary(details, passed, rank))

    if not ranked_results:
        if errors:
            raise ValueError("; ".join(errors))
        raise FileNotFoundError("No usable reference candidates were found for inspection.")

    _, passed, best_details = max(ranked_results, key=lambda item: item[0])
    if errors:
        best_details["reference_candidate_errors"] = errors
    best_details["evaluated_reference_ids"] = evaluated_ids
    best_details["reference_candidate_summaries"] = candidate_summaries
    return passed, best_details
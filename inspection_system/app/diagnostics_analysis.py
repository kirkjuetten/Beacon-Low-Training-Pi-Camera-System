#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter

from inspection_system.app.result_status import CONFIG_ERROR, FAIL, INVALID_CAPTURE, PASS, REGISTRATION_FAILED


GATE_SPECS = [
    {
        "code": "required_coverage",
        "title": "required coverage",
        "metric_key": "required_coverage",
        "threshold_keys": ("effective_min_required_coverage", "min_required_coverage"),
        "direction": "higher",
        "unit": "ratio",
        "false_reject_recommendation": "Lower min_required_coverage slightly, or expand approved-good reference variation if known-good parts are failing coverage.",
        "false_accept_recommendation": "Raise min_required_coverage if reject parts only pass because overall coverage is still being accepted.",
    },
    {
        "code": "outside_allowed_ratio",
        "title": "outside allowed ratio",
        "metric_key": "outside_allowed_ratio",
        "threshold_keys": ("effective_max_outside_allowed_ratio", "max_outside_allowed_ratio"),
        "direction": "lower",
        "unit": "ratio",
        "false_reject_recommendation": "Raise max_outside_allowed_ratio if acceptable cosmetic spill or lighting noise is being treated as a defect.",
        "false_accept_recommendation": "Lower max_outside_allowed_ratio if reject parts pass with extra material or contamination outside allowed regions.",
    },
    {
        "code": "min_section_coverage",
        "title": "section coverage",
        "metric_key": "min_section_coverage",
        "threshold_keys": ("effective_min_section_coverage", "min_section_coverage_limit", "min_section_coverage"),
        "direction": "higher",
        "unit": "ratio",
        "false_reject_recommendation": "Lower min_section_coverage if acceptable local variation is being rejected, or add more approved-good variants for molded-part spread.",
        "false_accept_recommendation": "Raise min_section_coverage if local missing-feature defects are slipping through.",
    },
    {
        "code": "mean_edge_distance_px",
        "title": "mean edge distance",
        "metric_key": "mean_edge_distance_px",
        "threshold_keys": ("effective_max_mean_edge_distance_px", "max_mean_edge_distance_px"),
        "direction": "lower",
        "unit": "px",
        "gate_key": "edge_distance_gate_active",
        "false_reject_recommendation": "Relax the mean edge distance gate or recommission with more approved-good shape variation if good parts show consistent edge drift.",
        "false_accept_recommendation": "Tighten max_mean_edge_distance_px if reject parts pass despite measurable overall edge-shape drift.",
    },
    {
        "code": "worst_section_edge_distance_px",
        "title": "section edge distance",
        "metric_key": "worst_section_edge_distance_px",
        "threshold_keys": ("effective_max_section_edge_distance_px", "max_section_edge_distance_px"),
        "direction": "lower",
        "unit": "px",
        "gate_key": "section_edge_gate_active",
        "false_reject_recommendation": "Relax the per-section edge gate or expand the approved-good geometry library if good parts fail on local edge drift.",
        "false_accept_recommendation": "Tighten max_section_edge_distance_px if localized edge-shape defects are passing.",
    },
    {
        "code": "worst_section_width_delta_ratio",
        "title": "section width drift",
        "metric_key": "worst_section_width_delta_ratio",
        "threshold_keys": ("effective_max_section_width_delta_ratio", "max_section_width_delta_ratio"),
        "direction": "lower",
        "unit": "ratio",
        "gate_key": "section_width_gate_active",
        "false_reject_recommendation": "Relax the section-width gate if acceptable molded-part width variation is failing inspection.",
        "false_accept_recommendation": "Tighten max_section_width_delta_ratio if width-related defects are slipping through.",
    },
    {
        "code": "worst_section_center_offset_px",
        "title": "section center offset",
        "metric_key": "worst_section_center_offset_px",
        "threshold_keys": ("effective_max_section_center_offset_px", "max_section_center_offset_px"),
        "direction": "lower",
        "unit": "px",
        "gate_key": "section_center_gate_active",
        "false_reject_recommendation": "Relax the section-center gate or improve fixturing/alignment if good loads are consistently offset.",
        "false_accept_recommendation": "Tighten max_section_center_offset_px if position-related defects are passing.",
    },
    {
        "code": "ssim",
        "title": "structural similarity",
        "metric_key": "ssim",
        "threshold_keys": ("effective_min_ssim", "min_ssim"),
        "direction": "higher",
        "unit": "score",
        "gate_key": "ssim_gate_active",
        "false_reject_recommendation": "Lower min_ssim if acceptable appearance variation is being rejected, or keep SSIM off for projects where appearance drift is not a real defect.",
        "false_accept_recommendation": "Raise min_ssim if obvious visual mismatches are still passing with SSIM enabled.",
    },
    {
        "code": "mse",
        "title": "mean squared error",
        "metric_key": "mse",
        "threshold_keys": ("effective_max_mse", "max_mse"),
        "direction": "lower",
        "unit": "score",
        "gate_key": "mse_gate_active",
        "false_reject_recommendation": "Raise max_mse if acceptable appearance drift is causing false rejects.",
        "false_accept_recommendation": "Lower max_mse if visible appearance defects are passing.",
    },
    {
        "code": "anomaly_score",
        "title": "anomaly score",
        "metric_key": "anomaly_score",
        "threshold_keys": ("effective_min_anomaly_score", "min_anomaly_score"),
        "direction": "higher",
        "unit": "score",
        "gate_key": "anomaly_gate_active",
        "false_reject_recommendation": "Collect more approved-good samples and rebuild the anomaly model before tightening anomaly gating further; lower min_anomaly_score only after confirming score overlap.",
        "false_accept_recommendation": "Raise min_anomaly_score if reject parts are nearly failing the anomaly gate, otherwise the defect may need a dedicated feature.",
    },
]

CLOSE_MARGIN_THRESHOLDS = {
    "required_coverage": 0.03,
    "outside_allowed_ratio": 0.01,
    "min_section_coverage": 0.03,
    "mean_edge_distance_px": 1.0,
    "worst_section_edge_distance_px": 1.0,
    "worst_section_width_delta_ratio": 0.02,
    "worst_section_center_offset_px": 1.0,
    "ssim": 0.03,
    "mse": 5.0,
    "anomaly_score": 0.05,
}


def _optional_float(value):
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_delta(value: float, unit: str) -> str:
    magnitude = abs(float(value))
    if unit == "ratio":
        return f"{magnitude:.1%}"
    if unit == "px":
        return f"{magnitude:.2f}px"
    return f"{magnitude:.3f}"


def _resolve_gate_threshold(details: dict, spec: dict) -> float | None:
    for key in spec.get("threshold_keys", ()):
        threshold = _optional_float(details.get(key))
        if threshold is not None:
            return threshold
    return None


def _is_gate_active(details: dict, spec: dict) -> bool:
    gate_key = spec.get("gate_key")
    if gate_key is None:
        return True
    if gate_key in details:
        return bool(details.get(gate_key))
    return _resolve_gate_threshold(details, spec) is not None


def _compute_gate_margin(details: dict, spec: dict) -> float | None:
    metric_value = _optional_float(details.get(spec["metric_key"]))
    threshold_value = _resolve_gate_threshold(details, spec)
    if metric_value is None or threshold_value is None:
        return None
    if spec["direction"] == "higher":
        return round(metric_value - threshold_value, 6)
    return round(threshold_value - metric_value, 6)


def _build_gate_entry(details: dict, spec: dict, margin: float) -> dict:
    metric_value = _optional_float(details.get(spec["metric_key"]))
    threshold_value = _resolve_gate_threshold(details, spec)
    failure = margin < 0
    direction_word = "below" if spec["direction"] == "higher" else "above"
    relation = "fails" if failure else "passes"
    summary = (
        f"{spec['title'].title()} {relation} by {_format_delta(margin, spec['unit'])}"
        if not failure
        else f"{spec['title'].title()} is {direction_word} threshold by {_format_delta(margin, spec['unit'])}"
    )
    return {
        "cause_code": spec["code"],
        "title": spec["title"],
        "metric_key": spec["metric_key"],
        "margin": round(margin, 6),
        "value": metric_value,
        "threshold": threshold_value,
        "summary": summary,
    }


def _alignment_warnings(details: dict) -> list[str]:
    warnings = []
    angle_correction = abs(float(details.get("best_angle_deg", 0.0) or 0.0))
    shift_x = float(details.get("best_shift_x", 0.0) or 0.0)
    shift_y = float(details.get("best_shift_y", 0.0) or 0.0)
    shift_distance = (shift_x * shift_x + shift_y * shift_y) ** 0.5
    if angle_correction > 0.5:
        warnings.append(f"Alignment rotation correction was {angle_correction:.2f} degrees")
    if shift_distance > 2.0:
        warnings.append(f"Alignment shift was {shift_distance:.2f} pixels")
    return warnings


def _build_feature_position_failure_mode(result: dict) -> dict | None:
    cause_code = str(result.get("inspection_failure_cause") or "").strip().lower()
    if cause_code not in {"feature_position", "feature_not_found", "light_pipe_position", "light_pipe_not_found"}:
        return None

    feature_summary = result.get("feature_position_summary", {})
    if not isinstance(feature_summary, dict):
        feature_summary = {}

    feature_label = str(feature_summary.get("feature_label") or feature_summary.get("feature_family") or "feature").replace("_", " ").title()
    radial_offset_px = _optional_float(feature_summary.get("radial_offset_px"))
    dx_px = _optional_float(feature_summary.get("dx_px"))
    dy_px = _optional_float(feature_summary.get("dy_px"))
    center_offset_px = _optional_float(feature_summary.get("center_offset_px"))
    pair_spacing_delta_px = _optional_float(feature_summary.get("pair_spacing_delta_px"))
    threshold = _resolve_gate_threshold(result, next(spec for spec in GATE_SPECS if spec["code"] == "worst_section_center_offset_px"))

    if cause_code.endswith("not_found"):
        summary = f"{feature_label} was not found in the expected datum window"
    elif dx_px is not None and dy_px is not None:
        summary = f"{feature_label} delta was dx={dx_px:+.2f}px, dy={dy_px:+.2f}px"
        if radial_offset_px is not None:
            summary += f" (radial {radial_offset_px:.2f}px)"
        if pair_spacing_delta_px is not None:
            summary += f", spacing delta {pair_spacing_delta_px:.2f}px"
    elif center_offset_px is not None and threshold is not None:
        summary = f"{feature_label} center offset was {center_offset_px:.2f}px against limit {threshold:.2f}px"
    else:
        summary = f"{feature_label} position failed the datum-position lane"

    return {
        "cause_code": cause_code,
        "title": f"{feature_label.lower()} position",
        "metric_key": "feature_position_summary",
        "margin": None,
        "value": radial_offset_px if radial_offset_px is not None else center_offset_px,
        "threshold": threshold,
        "summary": summary,
    }


def _active_lane_id(result: dict) -> str | None:
    inspection_program = result.get("inspection_program", {})
    if isinstance(inspection_program, dict) and inspection_program.get("active_lane_id"):
        return str(inspection_program.get("active_lane_id"))
    return None


def _primary_lane_id(result: dict) -> str | None:
    inspection_program = result.get("inspection_program", {})
    if isinstance(inspection_program, dict) and inspection_program.get("primary_lane_id"):
        return str(inspection_program.get("primary_lane_id"))
    return None


def _build_recommendation(priority: str, category: str, title: str, rationale: str, suggested_changes: list[str], evidence: dict) -> dict:
    return {
        "priority": priority,
        "category": category,
        "title": title,
        "rationale": rationale,
        "suggested_changes": suggested_changes,
        "evidence": evidence,
    }


def _is_close_pass_margin(entry: dict) -> bool:
    threshold = CLOSE_MARGIN_THRESHOLDS.get(entry.get("cause_code"), 0.03)
    return float(entry.get("margin", 999.0)) <= float(threshold)


def _diagnose_result(record: dict, result: dict) -> dict:
    status = str(result.get("status", "UNKNOWN"))
    if status == INVALID_CAPTURE:
        reason = result.get("reason", "Invalid capture")
        return {
            "primary_cause": "invalid_capture",
            "summary": str(reason),
            "failure_modes": [],
            "closest_pass_gates": [],
            "alignment_warnings": [],
        }
    if status == CONFIG_ERROR:
        reason = result.get("reason", "Configuration error")
        return {
            "primary_cause": "config_error",
            "summary": str(reason),
            "failure_modes": [],
            "closest_pass_gates": [],
            "alignment_warnings": [],
        }
    if status == REGISTRATION_FAILED:
        registration_reason = result.get("registration_rejection_reason") or result.get("reason") or "Registration failed"
        alignment_warnings = _alignment_warnings(result)
        summary_parts = [str(registration_reason)]
        if alignment_warnings:
            summary_parts.append(alignment_warnings[0])
        return {
            "primary_cause": "registration_failure",
            "summary": "; ".join(summary_parts),
            "failure_modes": list(result.get("registration_quality_gate_failures", [])),
            "closest_pass_gates": [],
            "alignment_warnings": alignment_warnings,
        }

    feature_failure_mode = _build_feature_position_failure_mode(result)
    if status == FAIL and feature_failure_mode is not None:
        alignment_warnings = _alignment_warnings(result)
        summary_parts = [feature_failure_mode["summary"]]
        if alignment_warnings:
            summary_parts.append(alignment_warnings[0])
        return {
            "primary_cause": feature_failure_mode["cause_code"],
            "summary": "; ".join(summary_parts),
            "failure_modes": [feature_failure_mode],
            "closest_pass_gates": [],
            "alignment_warnings": alignment_warnings,
        }

    failure_modes = []
    closest_pass_gates = []
    for spec in GATE_SPECS:
        if not _is_gate_active(result, spec):
            continue
        margin = _compute_gate_margin(result, spec)
        if margin is None:
            continue
        entry = _build_gate_entry(result, spec, margin)
        if margin < 0:
            failure_modes.append(entry)
        else:
            closest_pass_gates.append(entry)

    closest_pass_gates.sort(key=lambda entry: entry["margin"])
    alignment_warnings = _alignment_warnings(result)

    if status == PASS:
        summary_parts = ["Sample passed inspection"]
        if closest_pass_gates:
            nearest = closest_pass_gates[0]
            summary_parts.append(
                f"closest active gate was {nearest['title']} with {_format_delta(nearest['margin'], next(spec['unit'] for spec in GATE_SPECS if spec['code'] == nearest['cause_code']))} margin"
            )
        if alignment_warnings:
            summary_parts.append(alignment_warnings[0])
        return {
            "primary_cause": None,
            "summary": "; ".join(summary_parts),
            "failure_modes": [],
            "closest_pass_gates": closest_pass_gates[:3],
            "alignment_warnings": alignment_warnings,
        }

    if failure_modes:
        failure_modes.sort(key=lambda entry: entry["margin"])
        primary = failure_modes[0]["cause_code"]
        summary = "; ".join([entry["summary"] for entry in failure_modes[:2]] + alignment_warnings[:1])
    else:
        primary = "unknown_failure"
        summary = "Sample failed, but no active gate clearly explained the failure"
        if alignment_warnings:
            summary = "; ".join([summary, alignment_warnings[0]])

    return {
        "primary_cause": primary,
        "summary": summary,
        "failure_modes": failure_modes,
        "closest_pass_gates": closest_pass_gates[:3],
        "alignment_warnings": alignment_warnings,
    }


def _build_episode_analysis(training_report: dict, evaluation_report: dict) -> dict:
    results = evaluation_report.get("results", [])
    false_rejects = [result for result in results if result.get("expected_status") == PASS and result.get("actual_status") != PASS]
    false_accepts = [result for result in results if result.get("expected_status") == FAIL and result.get("actual_status") == PASS]
    invalid_capture_misses = [
        result for result in results if result.get("expected_status") == INVALID_CAPTURE and result.get("actual_status") != INVALID_CAPTURE
    ]

    false_reject_cause_counts = Counter(
        result.get("diagnosis", {}).get("primary_cause") or "unknown_failure"
        for result in false_rejects
    )
    false_reject_lane_counts = Counter(
        lane_id
        for result in false_rejects
        for lane_id in [_active_lane_id(result.get("result", {}))]
        if lane_id is not None
    )
    false_reject_patterns = [
        {
            "cause_code": cause_code,
            "count": count,
            "sample_images": [result["image_path"] for result in false_rejects if result.get("diagnosis", {}).get("primary_cause") == cause_code][:5],
        }
        for cause_code, count in false_reject_cause_counts.most_common()
    ]

    false_accept_category_counts = Counter(result.get("defect_category") or "unlabeled_reject" for result in false_accepts)
    false_accept_lane_counts = Counter(
        lane_id
        for result in false_accepts
        for lane_id in [_primary_lane_id(result.get("result", {}))]
        if lane_id is not None
    )
    false_accept_near_gate_counts = Counter()
    feature_gap_categories = Counter()
    for result in false_accepts:
        closest_gate = None
        for gate_entry in result.get("diagnosis", {}).get("closest_pass_gates", []):
            if _is_close_pass_margin(gate_entry):
                closest_gate = gate_entry
                break
        if closest_gate is None:
            feature_gap_categories[result.get("defect_category") or "unlabeled_reject"] += 1
        else:
            false_accept_near_gate_counts[closest_gate["cause_code"]] += 1

    invalid_capture_reasons = Counter(
        result.get("diagnosis", {}).get("summary") or result.get("actual_status") for result in invalid_capture_misses
    )
    registration_status_counts = Counter(
        str(result.get("result", {}).get("registration_status") or "unknown")
        for result in results
        if result.get("result", {}).get("registration_status")
    )
    registration_rejection_reason_counts = Counter(
        str(result.get("result", {}).get("registration_rejection_reason"))
        for result in results
        if result.get("result", {}).get("registration_rejection_reason")
    )

    recommendations = []
    spec_by_code = {spec["code"]: spec for spec in GATE_SPECS}
    for cause_code, count in false_reject_cause_counts.most_common(2):
        if cause_code == "registration_failure":
            recommendations.append(
                _build_recommendation(
                    "high" if count == len(false_rejects) and count > 0 else "medium",
                    "registration_review",
                    "Review registration quality gates",
                    f"{count} expected-good images failed during registration before part-level gates could be trusted.",
                    [
                        "Review anchor placement or registration mode if samples are not localizing consistently.",
                        "Relax registration quality gates only after confirming the transform is otherwise stable on approved-good samples.",
                    ],
                    {"false_reject_count": count, "cause_code": cause_code},
                )
            )
            continue
        if cause_code in {"feature_position", "feature_not_found", "light_pipe_position", "light_pipe_not_found"}:
            recommendations.append(
                _build_recommendation(
                    "high" if count == len(false_rejects) and count > 0 else "medium",
                    "feature_position_tuning",
                    "Review molded-part feature lane",
                    f"{count} expected-good images failed on an explicit molded-part feature-position lane.",
                    [
                        "Review the localized feature family selection and confirm the reference components represent the intended molded-part landmarks.",
                        "Tune the center-offset tolerance only after confirming the extractor is tracking the correct isolated or paired geometry.",
                    ],
                    {"false_reject_count": count, "cause_code": cause_code},
                )
            )
            continue
        spec = spec_by_code.get(cause_code)
        if spec is None:
            continue
        recommendations.append(
            _build_recommendation(
                "high" if count == len(false_rejects) and count > 0 else "medium",
                "threshold_relaxation",
                f"Review {spec['title']} false rejects",
                f"{count} expected-good images failed mainly on {spec['title']}.",
                [spec["false_reject_recommendation"]],
                {"false_reject_count": count, "cause_code": cause_code},
            )
        )

    if false_reject_lane_counts:
        lane_id, count = false_reject_lane_counts.most_common(1)[0]
        recommendations.append(
            _build_recommendation(
                "medium",
                "lane_tuning",
                f"Review {lane_id} lane false rejects",
                f"{count} expected-good images failed in active lane '{lane_id}'.",
                [
                    "Replay the saved challenge set filtered to that lane and inspect which thresholds or localized features are driving the failures.",
                    "Tune lane-specific thresholds before adjusting unrelated gates globally.",
                ],
                {"lane_id": lane_id, "false_reject_count": count},
            )
        )

    for cause_code, count in false_accept_near_gate_counts.most_common(2):
        spec = spec_by_code.get(cause_code)
        if spec is None:
            continue
        recommendations.append(
            _build_recommendation(
                "high" if count > 1 else "medium",
                "threshold_tightening",
                f"Tighten {spec['title']} for reject escapes",
                f"{count} expected-reject images passed close to the {spec['title']} gate.",
                [spec["false_accept_recommendation"]],
                {"false_accept_count": count, "cause_code": cause_code},
            )
        )

    if feature_gap_categories:
        top_categories = [category for category, _count in feature_gap_categories.most_common(3)]
        recommendations.append(
            _build_recommendation(
                "high",
                "feature_gap",
                "Add defect-specific features for reject escapes",
                "Some reject categories pass with comfortable margins on all active gates, which suggests the current feature set does not see those defects.",
                [
                    "Add region-specific or defect-specific measurements for the dominant escaping defect categories.",
                    "Review whether SSIM, ML gating, or new localized geometry checks should be enabled for those categories.",
                ],
                {"defect_categories": top_categories, "false_accept_count": sum(feature_gap_categories.values())},
            )
        )

    if false_accept_lane_counts:
        lane_id, count = false_accept_lane_counts.most_common(1)[0]
        recommendations.append(
            _build_recommendation(
                "medium",
                "lane_escape_review",
                f"Review reject escapes against {lane_id} lane",
                f"{count} expected-reject images still passed through primary lane '{lane_id}'.",
                [
                    "Inspect saved reject images lane-by-lane to confirm whether that lane needs a tighter gate or an additional localized feature.",
                    "Confirm the primary lane for those inspections matches the intended defect scope.",
                ],
                {"lane_id": lane_id, "false_accept_count": count},
            )
        )

    if invalid_capture_misses:
        recommendations.append(
            _build_recommendation(
                "high",
                "invalid_capture",
                "Improve invalid-capture detection",
                f"{len(invalid_capture_misses)} expected invalid captures were treated as inspectable images.",
                [
                    "Add or tighten blur, glare, clipped-frame, and part-present capture quality checks.",
                    "Review ROI bounds, exposure stability, and seat detection before normal inspection gating.",
                ],
                {"invalid_capture_miss_count": len(invalid_capture_misses), "top_reasons": dict(invalid_capture_reasons)},
            )
        )

    commissioning_status = training_report.get("commissioning_status", {})
    if not commissioning_status.get("ready", False):
        recommendations.append(
            _build_recommendation(
                "high",
                "commissioning",
                "Finish commissioning before trusting training results",
                commissioning_status.get("warning") or commissioning_status.get("summary_line") or "Commissioning is not ready.",
                commissioning_status.get("actions") or [commissioning_status.get("workflow_instruction", "Review the commissioning steps for this project.")],
                {
                    "workflow_stage": commissioning_status.get("workflow_stage_title"),
                    "committed_good_records": commissioning_status.get("committed_good_records"),
                    "pending_good_records": commissioning_status.get("pending_good_records"),
                },
            )
        )

    if not recommendations:
        recommendations.append(
            _build_recommendation(
                "low",
                "stability",
                "No dominant diagnostic issue detected",
                "The current episode did not reveal a single dominant failure mode.",
                ["Review per-image results for mixed or low-sample-size issues before changing thresholds."],
                {},
            )
        )

    high_level_findings = []
    if false_reject_patterns:
        top_false_reject = false_reject_patterns[0]
        high_level_findings.append(f"Top false-reject driver: {top_false_reject['cause_code']} ({top_false_reject['count']} images).")
    if false_reject_lane_counts:
        lane_id, count = false_reject_lane_counts.most_common(1)[0]
        high_level_findings.append(f"Top false-reject lane: {lane_id} ({count} images).")
    if false_accept_category_counts:
        category, count = false_accept_category_counts.most_common(1)[0]
        high_level_findings.append(f"Top reject escape category: {category} ({count} images).")
    if false_accept_lane_counts:
        lane_id, count = false_accept_lane_counts.most_common(1)[0]
        high_level_findings.append(f"Top reject escape lane: {lane_id} ({count} images).")
    if invalid_capture_misses:
        high_level_findings.append(f"Invalid-capture misses: {len(invalid_capture_misses)} images.")
    if registration_status_counts:
        status, count = registration_status_counts.most_common(1)[0]
        high_level_findings.append(f"Most common registration status: {status} ({count} images).")
    if not high_level_findings:
        high_level_findings.append("No dominant evaluation failure patterns were detected in this episode.")

    return {
        "high_level_findings": high_level_findings,
        "false_reject_patterns": false_reject_patterns,
        "lane_patterns": {
            "false_rejects_by_active_lane": dict(false_reject_lane_counts),
            "false_accepts_by_primary_lane": dict(false_accept_lane_counts),
        },
        "false_accept_patterns": {
            "by_defect_category": dict(false_accept_category_counts),
            "near_gate_counts": dict(false_accept_near_gate_counts),
            "feature_gap_categories": dict(feature_gap_categories),
        },
        "registration_patterns": {
            "status_counts": dict(registration_status_counts),
            "rejection_reasons": dict(registration_rejection_reason_counts),
        },
        "invalid_capture_patterns": dict(invalid_capture_reasons),
        "recommendations": recommendations,
    }
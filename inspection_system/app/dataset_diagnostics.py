#!/usr/bin/env python3
"""Diagnostic training episode runner for collected test-photo sessions."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from inspection_system.app.camera_interface import get_active_runtime_paths, load_config
from inspection_system.app.interactive_training import ThresholdTrainer
from inspection_system.app.reference_service import clear_anomaly_training_artifacts, clear_reference_variants
from inspection_system.app.replay_inspection import inspect_file
from inspection_system.app.result_status import CONFIG_ERROR, FAIL, INVALID_CAPTURE, PASS, REGISTRATION_FAILED
from inspection_system.app.runtime_controller import get_commissioning_status, load_anomaly_detector

EPISODE_STATUS = {"good": PASS, "reject": FAIL, "invalid_capture": INVALID_CAPTURE}
TRAINING_BUCKETS = {"good", "reject"}
EVALUATION_BUCKETS = {"good", "reject", "borderline", "invalid_capture"}
THRESHOLD_KEYS = [
    "min_required_coverage",
    "max_outside_allowed_ratio",
    "min_section_coverage",
    "max_mean_edge_distance_px",
    "max_section_edge_distance_px",
    "max_section_width_delta_ratio",
    "max_section_center_offset_px",
    "min_ssim",
    "max_mse",
    "min_anomaly_score",
]

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


def _format_value(value, unit: str) -> str:
    if value is None:
        return "n/a"
    numeric_value = float(value)
    if unit == "ratio":
        return f"{numeric_value:.1%}"
    if unit == "px":
        return f"{numeric_value:.2f}px"
    return f"{numeric_value:.3f}"


def _resolve_gate_threshold(details: dict, spec: dict) -> float | None:
    for key in spec.get("threshold_keys", ()):  # pragma: no branch - short list
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


def _is_close_pass_margin(entry: dict) -> bool:
    threshold = CLOSE_MARGIN_THRESHOLDS.get(entry.get("cause_code"), 0.03)
    return float(entry.get("margin", 999.0)) <= float(threshold)


def _build_recommendation(priority: str, category: str, title: str, rationale: str, suggested_changes: list[str], evidence: dict) -> dict:
    return {
        "priority": priority,
        "category": category,
        "title": title,
        "rationale": rationale,
        "suggested_changes": suggested_changes,
        "evidence": evidence,
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
    false_reject_patterns = [
        {
            "cause_code": cause_code,
            "count": count,
            "sample_images": [result["image_path"] for result in false_rejects if result.get("diagnosis", {}).get("primary_cause") == cause_code][:5],
        }
        for cause_code, count in false_reject_cause_counts.most_common()
    ]

    false_accept_category_counts = Counter(result.get("defect_category") or "unlabeled_reject" for result in false_accepts)
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
        high_level_findings.append(
            f"Top false-reject driver: {top_false_reject['cause_code']} ({top_false_reject['count']} images)."
        )
    if false_accept_category_counts:
        category, count = false_accept_category_counts.most_common(1)[0]
        high_level_findings.append(f"Top reject escape category: {category} ({count} images).")
    if invalid_capture_misses:
        high_level_findings.append(f"Invalid-capture misses: {len(invalid_capture_misses)} images.")
    if not high_level_findings:
        high_level_findings.append("No dominant evaluation failure patterns were detected in this episode.")

    return {
        "high_level_findings": high_level_findings,
        "false_reject_patterns": false_reject_patterns,
        "false_accept_patterns": {
            "by_defect_category": dict(false_accept_category_counts),
            "near_gate_counts": dict(false_accept_near_gate_counts),
            "feature_gap_categories": dict(feature_gap_categories),
        },
        "invalid_capture_patterns": dict(invalid_capture_reasons),
        "recommendations": recommendations,
    }


def resolve_capture_manifests(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    direct = path / "captures.jsonl"
    if direct.exists():
        return [direct]
    return sorted(path.rglob("captures.jsonl"))


def load_capture_records(path: Path) -> list[dict]:
    manifests = resolve_capture_manifests(path)
    records: list[dict] = []
    for manifest_path in manifests:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["manifest_path"] = str(manifest_path)
                session_dir = manifest_path.parent
                record.setdefault("session_dir", str(session_dir))
                image_path = record.get("image_path")
                if image_path:
                    candidate_path = Path(str(image_path))
                    if not candidate_path.exists():
                        relative_image_path = record.get("relative_image_path")
                        if relative_image_path:
                            local_image_path = session_dir / str(relative_image_path)
                            if local_image_path.exists():
                                record["image_path"] = str(local_image_path)
                records.append(record)
    return records


def expected_status_for_record(record: dict) -> str | None:
    explicit = record.get("expected_inspection_status")
    if explicit:
        return str(explicit)
    return EPISODE_STATUS.get(str(record.get("bucket", "")).strip().lower())


def partition_episode_records(
    records: list[dict],
    *,
    train_splits: tuple[str, ...] = ("tuning",),
    eval_splits: tuple[str, ...] = ("validation", "regression"),
) -> tuple[list[dict], list[dict], dict]:
    train_candidates = [
        record for record in records
        if str(record.get("bucket", "")).strip().lower() in TRAINING_BUCKETS
        and str(record.get("dataset_split", "")).strip().lower() in train_splits
    ]
    if not train_candidates:
        train_candidates = [
            record for record in records if str(record.get("bucket", "")).strip().lower() in TRAINING_BUCKETS
        ]

    eval_candidates = [
        record for record in records
        if str(record.get("bucket", "")).strip().lower() in EVALUATION_BUCKETS
        and str(record.get("dataset_split", "")).strip().lower() in eval_splits
    ]
    reused_training_for_eval = False
    if not eval_candidates:
        eval_candidates = [
            record for record in records if str(record.get("bucket", "")).strip().lower() in EVALUATION_BUCKETS
        ]
        reused_training_for_eval = True

    partition_info = {
        "training_unique_count": len(train_candidates),
        "evaluation_unique_count": len(eval_candidates),
        "evaluation_reused_training_pool": reused_training_for_eval,
    }
    return train_candidates, eval_candidates, partition_info


def duplicate_training_records(records: list[dict], duplicate_count: int, shuffle_seed: int | None = None) -> list[dict]:
    duplicate_count = max(1, int(duplicate_count))
    expanded: list[dict] = []
    for cycle_index in range(duplicate_count):
        for record in records:
            expanded.append({**record, "simulation_cycle": cycle_index + 1})
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(expanded)
    return expanded


def build_diagnostic_output_path(source_path: Path, output_path: Path | None = None) -> Path:
    if output_path is not None:
        return output_path
    base_dir = source_path if source_path.is_dir() else source_path.parent
    diagnostics_dir = base_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return diagnostics_dir / f"episode_{time.strftime('%Y%m%d_%H%M%S')}.json"


def resolve_project_root_from_source(source_path: Path) -> Path | None:
    current = source_path if source_path.is_dir() else source_path.parent
    for candidate in [current, *current.parents]:
        config_file = candidate / "config" / "camera_config.json"
        reference_dir = candidate / "reference"
        if config_file.exists() and reference_dir.exists():
            return candidate
        if candidate.name == "test_data":
            project_root = candidate.parent
            config_file = project_root / "config" / "camera_config.json"
            reference_dir = project_root / "reference"
            if config_file.exists() and reference_dir.exists():
                return project_root
    return None


def build_active_paths_from_project_root(project_root: Path) -> dict:
    reference_dir = project_root / "reference"
    return {
        "config_file": project_root / "config" / "camera_config.json",
        "log_dir": project_root / "logs",
        "reference_dir": reference_dir,
        "reference_mask": reference_dir / "golden_reference_mask.png",
        "reference_image": reference_dir / "golden_reference_image.png",
    }


def resolve_project_context(source_path: Path) -> tuple[dict, dict, Path | None]:
    project_root = resolve_project_root_from_source(source_path)
    if project_root is None:
        active_paths = get_active_runtime_paths()
        return load_config(), active_paths, None

    active_paths = build_active_paths_from_project_root(project_root)
    with Path(active_paths["config_file"]).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config, active_paths, project_root


def _copy_runtime_inputs(active_paths: dict, sandbox_root: Path) -> dict:
    config_dir = sandbox_root / "config"
    reference_dir = sandbox_root / "reference"
    log_dir = sandbox_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    source_config = Path(active_paths["config_file"])
    sandbox_config = config_dir / source_config.name
    shutil.copy2(source_config, sandbox_config)

    source_reference_dir = Path(active_paths["reference_dir"])
    if source_reference_dir.exists():
        for child in source_reference_dir.iterdir():
            target = reference_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)

    return {
        "config_file": sandbox_config,
        "reference_dir": reference_dir,
        "log_dir": log_dir,
        "reference_mask": reference_dir / "golden_reference_mask.png",
        "reference_image": reference_dir / "golden_reference_image.png",
    }


def _load_sandbox_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _snapshot_thresholds(config: dict) -> dict:
    inspection_cfg = config.get("inspection", {})
    return {key: inspection_cfg.get(key) for key in THRESHOLD_KEYS}


def _apply_training_update(trainer: ThresholdTrainer, config_state: dict) -> dict:
    pending_before = len(trainer.get_pending_records())
    pending_anomaly_samples = trainer.get_pending_anomaly_sample_count()
    suggestions = trainer.suggest_thresholds()
    learned_ranges = trainer.extract_learned_ranges()
    learning_update = trainer.apply_learning_update(config_state, suggestions, learned_ranges) if (suggestions or learned_ranges) else {
        "threshold_updates": {},
        "learned_ranges_saved": False,
        "learned_ranges_changed": False,
    }
    committed = trainer.commit_pending_feedback()
    rebuild_result = None
    inspection_mode = str(config_state.get("inspection", {}).get("inspection_mode", "mask_only")).strip().lower()
    if pending_anomaly_samples or inspection_mode in {"mask_and_ml", "full"}:
        rebuild_result = trainer.rebuild_anomaly_model(config_state)
    return {
        "pending_before_update": pending_before,
        "committed_count": committed,
        "threshold_updates": learning_update.get("threshold_updates", {}),
        "learned_ranges_saved": learning_update.get("learned_ranges_saved", False),
        "learned_ranges_changed": learning_update.get("learned_ranges_changed", False),
        "anomaly_rebuild": rebuild_result,
    }


def _evaluate_records(records: list[dict], config_state: dict, active_paths: dict) -> dict:
    actual_counts = Counter()
    expected_counts = Counter()
    confusion = defaultdict(Counter)
    borderline_counts = Counter()
    results = []

    false_reject_count = 0
    false_accept_count = 0
    invalid_capture_miss_count = 0

    for record in records:
        image_path = Path(record["image_path"])
        result = inspect_file(config_state, image_path, active_paths=active_paths)
        diagnosis = _diagnose_result(record, result)
        actual_status = str(result.get("status", "UNKNOWN"))
        bucket = str(record.get("bucket", "")).strip().lower()
        expected_status = expected_status_for_record(record)

        actual_counts[actual_status] += 1
        if expected_status:
            expected_counts[expected_status] += 1
            confusion[expected_status][actual_status] += 1
            if expected_status == PASS and actual_status != PASS:
                false_reject_count += 1
            elif expected_status == FAIL and actual_status == PASS:
                false_accept_count += 1
            elif expected_status == INVALID_CAPTURE and actual_status != INVALID_CAPTURE:
                invalid_capture_miss_count += 1
        elif bucket == "borderline":
            borderline_counts[actual_status] += 1

        results.append(
            {
                "image_path": str(image_path),
                "bucket": bucket,
                "dataset_split": record.get("dataset_split"),
                "defect_category": record.get("defect_category"),
                "note": record.get("note"),
                "expected_status": expected_status,
                "actual_status": actual_status,
                "diagnosis": diagnosis,
                "result": result,
            }
        )

    good_total = expected_counts[PASS]
    reject_total = expected_counts[FAIL]
    invalid_total = expected_counts[INVALID_CAPTURE]
    return {
        "evaluated_count": len(records),
        "expected_counts": dict(expected_counts),
        "actual_counts": dict(actual_counts),
        "confusion_matrix": {expected: dict(actuals) for expected, actuals in confusion.items()},
        "borderline_outcomes": dict(borderline_counts),
        "false_reject_count": false_reject_count,
        "false_accept_count": false_accept_count,
        "invalid_capture_miss_count": invalid_capture_miss_count,
        "false_reject_rate": round(false_reject_count / good_total, 4) if good_total else None,
        "false_accept_rate": round(false_accept_count / reject_total, 4) if reject_total else None,
        "invalid_capture_miss_rate": round(invalid_capture_miss_count / invalid_total, 4) if invalid_total else None,
        "results": results,
    }


def simulate_training_episode(
    source_path: Path,
    *,
    config: dict | None = None,
    active_paths: dict | None = None,
    duplicate_count: int = 1,
    update_every: int = 5,
    shuffle_seed: int | None = None,
    train_splits: tuple[str, ...] = ("tuning",),
    eval_splits: tuple[str, ...] = ("validation", "regression"),
) -> dict:
    records = load_capture_records(source_path)
    if not records:
        raise ValueError(f"No capture records found under {source_path}")

    runtime_paths = active_paths or get_active_runtime_paths()
    training_records, evaluation_records, partition_info = partition_episode_records(
        records,
        train_splits=train_splits,
        eval_splits=eval_splits,
    )
    duplicated_training_records = duplicate_training_records(training_records, duplicate_count, shuffle_seed)

    with tempfile.TemporaryDirectory(prefix="beacon_episode_") as sandbox_dir:
        sandbox_root = Path(sandbox_dir)
        sandbox_paths = _copy_runtime_inputs(runtime_paths, sandbox_root)
        clear_reference_variants(sandbox_paths)
        clear_anomaly_training_artifacts(sandbox_paths)

        trainer = ThresholdTrainer(Path(sandbox_paths["config_file"]), active_paths=sandbox_paths)
        config_state = _load_sandbox_config(Path(sandbox_paths["config_file"]))
        initial_thresholds = _snapshot_thresholds(config_state)

        update_every = max(1, int(update_every))
        update_events = []
        training_replay_mismatches = 0

        for index, record in enumerate(duplicated_training_records, start=1):
            image_path = Path(record["image_path"])
            replay_result = inspect_file(config_state, image_path, active_paths=sandbox_paths)
            if expected_status_for_record(record) and replay_result.get("status") != expected_status_for_record(record):
                training_replay_mismatches += 1

            bucket = str(record.get("bucket", "")).strip().lower()
            feedback = "approve" if bucket == "good" else "reject"
            trainer.record_feedback(
                replay_result,
                feedback,
                label_info={
                    "final_class": "good" if bucket == "good" else "reject",
                    "defect_category": record.get("defect_category"),
                    "classification_reason": record.get("note"),
                },
                image_path=image_path,
            )

            if index % update_every == 0:
                update_snapshot = _apply_training_update(trainer, config_state)
                update_snapshot["after_training_record"] = index
                update_events.append(update_snapshot)
                config_state = _load_sandbox_config(Path(sandbox_paths["config_file"]))

        if trainer.get_pending_records():
            update_snapshot = _apply_training_update(trainer, config_state)
            update_snapshot["after_training_record"] = len(duplicated_training_records)
            update_events.append(update_snapshot)
            config_state = _load_sandbox_config(Path(sandbox_paths["config_file"]))

        final_thresholds = _snapshot_thresholds(config_state)
        threshold_drift = {
            key: {"initial": initial_thresholds.get(key), "final": final_thresholds.get(key)}
            for key in THRESHOLD_KEYS
            if initial_thresholds.get(key) != final_thresholds.get(key)
        }
        anomaly_detector = load_anomaly_detector(sandbox_paths)
        commissioning_status = get_commissioning_status(config_state, sandbox_paths, anomaly_detector)
        training_report = {
            "processed_count": len(duplicated_training_records),
            "training_replay_mismatch_count": training_replay_mismatches,
            "update_events": update_events,
            "pending_after_run": trainer.get_pending_summary(),
            "learning_record_count": len(trainer.get_learning_records()),
            "threshold_drift": threshold_drift,
            "final_thresholds": final_thresholds,
            "commissioning_status": commissioning_status,
        }
        evaluation = _evaluate_records(evaluation_records, config_state, sandbox_paths)
        analysis = _build_episode_analysis(training_report, evaluation)

        return {
            "source_path": str(source_path),
            "episode_parameters": {
                "duplicate_count": int(duplicate_count),
                "update_every": update_every,
                "shuffle_seed": shuffle_seed,
                "train_splits": list(train_splits),
                "eval_splits": list(eval_splits),
            },
            "partition": partition_info,
            "training": training_report,
            "evaluation": evaluation,
            "analysis": analysis,
        }


def save_episode_report(report: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sandboxed diagnostic training episode from collected test-photo sessions.")
    parser.add_argument("source", type=Path, help="Session directory, captures.jsonl path, or test_data root")
    parser.add_argument("--duplicates", type=int, default=2, help="How many times to replay the training pool during the episode")
    parser.add_argument("--update-every", type=int, default=5, help="How many simulated training parts between Update actions")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle seed for duplicate replay ordering")
    parser.add_argument("--train-split", dest="train_splits", action="append", default=None, help="Dataset split(s) to use for training, default: tuning")
    parser.add_argument("--eval-split", dest="eval_splits", action="append", default=None, help="Dataset split(s) to use for evaluation, default: validation and regression")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the JSON diagnostics report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train_splits = tuple(args.train_splits or ["tuning"])
    eval_splits = tuple(args.eval_splits or ["validation", "regression"])
    resolved_config, resolved_active_paths, project_root = resolve_project_context(args.source)
    report = simulate_training_episode(
        args.source,
        config=resolved_config,
        active_paths=resolved_active_paths,
        duplicate_count=args.duplicates,
        update_every=args.update_every,
        shuffle_seed=args.seed,
        train_splits=train_splits,
        eval_splits=eval_splits,
    )
    output_path = build_diagnostic_output_path(args.source, args.output)
    save_episode_report(report, output_path)
    print(json.dumps({
        "report_path": str(output_path),
        "project_root": str(project_root) if project_root is not None else None,
        "training_processed": report["training"]["processed_count"],
        "false_reject_rate": report["evaluation"]["false_reject_rate"],
        "false_accept_rate": report["evaluation"]["false_accept_rate"],
        "invalid_capture_miss_rate": report["evaluation"]["invalid_capture_miss_rate"],
        "commissioning_ready": report["training"]["commissioning_status"].get("ready"),
        "top_recommendation": (report.get("analysis", {}).get("recommendations") or [{}])[0].get("title"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
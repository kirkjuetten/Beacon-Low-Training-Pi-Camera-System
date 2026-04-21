#!/usr/bin/env python3
"""Shared helpers for training record schema and fingerprint handling."""

from __future__ import annotations

from collections.abc import Callable

from inspection_system.app.segmentation import build_legacy_threshold_mode, resolve_segmentation_settings


def build_config_fingerprint(config: dict | None) -> dict:
    source = config or {}
    inspection_cfg = source.get("inspection", {}) if isinstance(source, dict) else {}
    alignment_cfg = source.get("alignment", {}) if isinstance(source, dict) else {}
    segmentation_settings = resolve_segmentation_settings(inspection_cfg)
    return {
        "inspection_mode": inspection_cfg.get("inspection_mode", "mask_only"),
        "reference_strategy": inspection_cfg.get("reference_strategy", "golden_only"),
        "blend_mode": inspection_cfg.get("blend_mode", "hard_only"),
        "tolerance_mode": inspection_cfg.get("tolerance_mode", "balanced"),
        "segmentation_strategy": segmentation_settings["strategy_name"],
        "threshold_method": segmentation_settings["threshold_method"],
        "threshold_mode": build_legacy_threshold_mode(segmentation_settings),
        "threshold_value": segmentation_settings["threshold_value"],
        "min_required_coverage": inspection_cfg.get("min_required_coverage"),
        "max_outside_allowed_ratio": inspection_cfg.get("max_outside_allowed_ratio"),
        "min_section_coverage": inspection_cfg.get("min_section_coverage"),
        "max_mean_edge_distance_px": inspection_cfg.get("max_mean_edge_distance_px"),
        "max_section_edge_distance_px": inspection_cfg.get("max_section_edge_distance_px"),
        "max_section_width_delta_ratio": inspection_cfg.get("max_section_width_delta_ratio"),
        "max_section_center_offset_px": inspection_cfg.get("max_section_center_offset_px"),
        "min_ssim": inspection_cfg.get("min_ssim"),
        "max_mse": inspection_cfg.get("max_mse"),
        "min_anomaly_score": inspection_cfg.get("min_anomaly_score"),
        "alignment_profile": alignment_cfg.get("tolerance_profile", "balanced"),
    }


def normalize_record_schema(
    record: dict,
    *,
    schema_version: int,
    default_final_class: Callable[[str], str | None],
    config_fingerprint: dict,
    timestamp_provider: Callable[[], float],
) -> bool:
    changed = False
    if record.get("schema_version") != schema_version:
        record["schema_version"] = schema_version
        changed = True
    if "final_class" not in record:
        record["final_class"] = default_final_class(record.get("feedback", ""))
        changed = True
    if "defect_category" not in record:
        record["defect_category"] = None
        changed = True
    if "classification_reason" not in record:
        record["classification_reason"] = None
        changed = True
    if "config_fingerprint" not in record:
        record["config_fingerprint"] = config_fingerprint
        changed = True
    if "record_id" not in record:
        record["record_id"] = f"legacy_{int(record.get('timestamp', timestamp_provider()) * 1000)}"
        changed = True
    if "reference_candidate_id" not in record:
        record["reference_candidate_id"] = None
        changed = True
    if "reference_candidate_state" not in record:
        record["reference_candidate_state"] = None
        changed = True
    if "anomaly_sample_id" not in record:
        record["anomaly_sample_id"] = None
        changed = True
    if "anomaly_sample_state" not in record:
        record["anomaly_sample_state"] = None
        changed = True
    return changed


def build_training_record(
    details: dict,
    feedback: str,
    *,
    schema_version: int,
    record_id: str,
    timestamp: float,
    final_class: str | None,
    label_info: dict,
    config_fingerprint: dict,
) -> dict:
    return {
        "schema_version": schema_version,
        "record_id": record_id,
        "timestamp": timestamp,
        "feedback": feedback,
        "final_class": final_class,
        "defect_category": label_info.get("defect_category"),
        "classification_reason": label_info.get("classification_reason"),
        "learning_state": "pending",
        "config_fingerprint": config_fingerprint,
        "reference_candidate_id": None,
        "reference_candidate_state": None,
        "anomaly_sample_id": None,
        "anomaly_sample_state": None,
        "metrics": {
            "required_coverage": details.get("required_coverage", 0),
            "outside_allowed_ratio": details.get("outside_allowed_ratio", 0),
            "min_section_coverage": details.get("min_section_coverage", 0),
            "mean_edge_distance_px": details.get("mean_edge_distance_px"),
            "worst_section_edge_distance_px": details.get("worst_section_edge_distance_px"),
            "worst_section_width_delta_ratio": details.get("worst_section_width_delta_ratio"),
            "worst_section_center_offset_px": details.get("worst_section_center_offset_px"),
            "ssim": details.get("ssim"),
            "mse": details.get("mse"),
            "anomaly_score": details.get("anomaly_score"),
            "histogram_similarity": details.get("histogram_similarity"),
            "best_angle_deg": details.get("best_angle_deg", 0),
            "best_shift_x": details.get("best_shift_x", 0),
            "best_shift_y": details.get("best_shift_y", 0),
            "inspection_mode": details.get("inspection_mode", "mask_only"),
        },
    }
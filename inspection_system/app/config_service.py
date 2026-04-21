#!/usr/bin/env python3
"""Shared config schema and read/write helpers for UI surfaces."""

from __future__ import annotations

import json
from pathlib import Path


CAPTURE_CONFIG_FIELD_SPECS = [
    ("capture.timeout_ms", "Capture Timeout (ms)", int),
    ("capture.shutter_us", "Shutter (us)", int),
]

INSPECTION_CONFIG_FIELD_SPECS = [
    ("inspection.inspection_mode", "Inspection Mode", str),
    ("inspection.reference_strategy", "Reference Strategy", str),
    ("inspection.blend_mode", "Blend Mode", str),
    ("inspection.tolerance_mode", "Tolerance Mode", str),
    ("inspection.segmentation_strategy", "Segmentation Strategy", str),
    ("inspection.threshold_method", "Threshold Method", str),
    ("inspection.threshold_mode", "Threshold Mode", str),
    ("inspection.threshold_value", "Threshold Value", int),
    ("inspection.blur_kernel", "Blur Kernel (pixels)", int),
    ("inspection.reference_erode_iterations", "Reference Erode Iterations", int),
    ("inspection.reference_dilate_iterations", "Reference Dilate Iterations", int),
    ("inspection.sample_erode_iterations", "Sample Erode Iterations", int),
    ("inspection.sample_dilate_iterations", "Sample Dilate Iterations", int),
    ("inspection.min_feature_pixels", "Min Feature Pixels", int),
    ("inspection.min_required_coverage", "Min Required Coverage", float),
    ("inspection.max_outside_allowed_ratio", "Max Outside Allowed", float),
    ("inspection.min_section_coverage", "Min Section Coverage", float),
    ("inspection.max_mean_edge_distance_px", "Max Mean Edge Distance (px, optional)", float),
    ("inspection.max_section_edge_distance_px", "Max Section Edge Distance (px, optional)", float),
    ("inspection.max_section_width_delta_ratio", "Max Section Width Drift (ratio, optional)", float),
    ("inspection.max_section_center_offset_px", "Max Section Center Offset (px, optional)", float),
    ("inspection.min_ssim", "Min SSIM (optional)", float),
    ("inspection.max_mse", "Max MSE (optional)", float),
    ("inspection.min_anomaly_score", "Min Anomaly Score (optional)", float),
    ("inspection.image_display_mode", "Image Display Mode", str),
    ("inspection.save_debug_images", "Save Debug Images", bool),
]

ALIGNMENT_CONFIG_FIELD_SPECS = [
    ("alignment.enabled", "Alignment Enabled", bool),
    ("alignment.mode", "Active Registration Runtime", str),
    ("alignment.tolerance_profile", "Alignment Profile", str),
]

REGISTRATION_CONFIG_FIELD_SPECS = [
    ("alignment.registration.strategy", "Registration Strategy", str),
    ("alignment.registration.transform_model", "Registration Transform Model", str),
    ("alignment.registration.anchor_mode", "Registration Anchor Mode", str),
    ("alignment.registration.subpixel_refinement", "Subpixel Refinement", str),
    ("alignment.registration.search_margin_px", "Registration Search Margin (px)", int),
    ("alignment.registration.quality_gates.min_confidence", "Registration Min Confidence (optional)", float),
    (
        "alignment.registration.quality_gates.max_mean_residual_px",
        "Registration Max Mean Residual (px, optional)",
        float,
    ),
    ("alignment.registration.datum_frame.origin", "Datum Frame Origin", str),
    ("alignment.registration.datum_frame.orientation", "Datum Frame Orientation", str),
]

INDICATOR_CONFIG_FIELD_SPECS = [
    ("indicator_led.enabled", "Indicator LED Enabled", bool),
]


CONFIG_FIELD_SPECS = (
    CAPTURE_CONFIG_FIELD_SPECS
    + INSPECTION_CONFIG_FIELD_SPECS
    + ALIGNMENT_CONFIG_FIELD_SPECS
    + REGISTRATION_CONFIG_FIELD_SPECS
    + INDICATOR_CONFIG_FIELD_SPECS
)

CONFIG_DROPDOWN_OPTIONS = {
    "inspection.inspection_mode": ["mask_only", "mask_and_ssim", "mask_and_ml", "full"],
    "inspection.reference_strategy": ["golden_only", "hybrid", "multi_good_experimental"],
    "inspection.blend_mode": ["hard_only", "blend_conservative", "blend_balanced", "blend_aggressive"],
    "inspection.tolerance_mode": ["strict", "balanced", "forgiving", "custom"],
    "inspection.segmentation_strategy": ["binary_threshold", "binary_threshold_inverted"],
    "inspection.threshold_method": ["fixed", "otsu"],
    "inspection.threshold_mode": ["fixed", "fixed_inv", "otsu", "otsu_inv"],
    "inspection.image_display_mode": ["raw", "processed", "split"],
    "inspection.save_debug_images": ["True", "False"],
    "alignment.enabled": ["True", "False"],
    "alignment.mode": ["moments", "anchor_translation", "anchor_pair", "rigid_refined"],
    "alignment.tolerance_profile": ["strict", "balanced", "forgiving"],
    "alignment.registration.strategy": ["moments", "anchor_translation", "anchor_pair", "rigid_refined"],
    "alignment.registration.transform_model": ["rigid", "similarity", "affine"],
    "alignment.registration.anchor_mode": ["none", "single", "pair", "multi"],
    "alignment.registration.subpixel_refinement": ["off", "phase_correlation", "template"],
    "alignment.registration.datum_frame.origin": ["roi_top_left", "anchor_primary", "part_centroid"],
    "alignment.registration.datum_frame.orientation": ["part_axis", "image_axes", "anchor_pair"],
    "indicator_led.enabled": ["True", "False"],
}

OPTIONAL_FLOAT_FIELDS = {
    "inspection.max_mean_edge_distance_px",
    "inspection.max_section_edge_distance_px",
    "inspection.max_section_width_delta_ratio",
    "inspection.max_section_center_offset_px",
    "inspection.min_ssim",
    "inspection.max_mse",
    "inspection.min_anomaly_score",
    "alignment.registration.quality_gates.min_confidence",
    "alignment.registration.quality_gates.max_mean_residual_px",
}


def read_json_file(file_path: Path) -> dict:
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json_file(file_path: Path, data: dict) -> None:
    file_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def get_nested_config_value(config: dict, dotted_path: str):
    current = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_nested_config_value(config: dict, dotted_path: str, value) -> None:
    parts = dotted_path.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def parse_config_value(raw_value: str, expected_type: type):
    text = raw_value.strip()
    if expected_type is bool:
        normalized = text.lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {raw_value}")
    if expected_type is int:
        return int(text)
    if expected_type is float:
        return float(text)
    return text


def apply_config_updates(config: dict, raw_updates: dict[str, str]) -> dict:
    updated = json.loads(json.dumps(config))
    for dotted_path, _, expected_type in CONFIG_FIELD_SPECS:
        if dotted_path not in raw_updates:
            continue
        raw_value = raw_updates[dotted_path]
        if dotted_path in OPTIONAL_FLOAT_FIELDS and raw_value.strip() == "":
            set_nested_config_value(updated, dotted_path, None)
            continue
        if raw_value.strip() == "":
            # Keep existing value unchanged when a field is blank in sparse legacy configs.
            continue
        set_nested_config_value(updated, dotted_path, parse_config_value(raw_value, expected_type))
    return updated


def build_config_editor_values(config: dict) -> dict[str, str]:
    values = {}
    for dotted_path, _, _ in CONFIG_FIELD_SPECS:
        value = get_nested_config_value(config, dotted_path)
        values[dotted_path] = "" if value is None else str(value)
    return values
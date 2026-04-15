from __future__ import annotations

import logging
from pathlib import Path
from inspection_system.app.anomaly_detection_utils import detect_anomalies
from inspection_system.app.reference_service import check_reference_settings_match


logger = logging.getLogger(__name__)


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
) -> tuple[bool, dict]:
    inspection_cfg = config.get("inspection", {})
    alignment_cfg, alignment_profile = resolve_alignment_config(config)
    
    # Check if reference settings match current config
    settings_match, mismatch_msg = check_reference_settings_match(config)
    if not settings_match:
        logger.warning(f"Reference settings mismatch: {mismatch_msg}")
    
    roi_image, gray, sample_mask, roi, cv2, np = make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy)

    sample_erode_iterations = int(inspection_cfg.get("sample_erode_iterations", 1))
    sample_dilate_iterations = int(inspection_cfg.get("sample_dilate_iterations", 1))
    sample_mask = erode_mask(sample_mask, sample_erode_iterations, cv2, np)
    sample_mask = dilate_mask(sample_mask, sample_dilate_iterations, cv2, np)

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

    # Compute anomaly metrics before alignment
    anomaly_metrics = detect_anomalies(roi_image, reference_image, sample_mask, anomaly_detector)

    aligned_sample_mask, best_angle_deg, best_shift_x, best_shift_y = align_sample_mask(
        sample_mask,
        reference_mask,
        alignment_cfg,
        cv2,
        np,
    )

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

    metrics = score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks)
    metric_inputs = {**metrics, **anomaly_metrics}
    passed, threshold_summary = evaluate_metrics(metric_inputs, inspection_cfg)

    required_coverage = float(threshold_summary["required_coverage"])
    outside_allowed_ratio = float(threshold_summary["outside_allowed_ratio"])
    min_section_coverage = float(threshold_summary["min_section_coverage"])

    min_required_coverage = float(threshold_summary["min_required_coverage"])
    max_outside_allowed_ratio = float(threshold_summary["max_outside_allowed_ratio"])
    min_section_coverage_limit = float(threshold_summary["min_section_coverage_limit"])

    required_white = reference_required > 0
    allowed_white = reference_allowed > 0

    # Check memory bounds before allocating large arrays
    image_size = aligned_sample_mask.shape[0] * aligned_sample_mask.shape[1] * 3
    max_reasonable_pixels = 50 * 1024 * 1024  # 50MP limit
    if image_size > max_reasonable_pixels:
        raise MemoryError(f"Image too large for processing: {image_size} pixels exceeds {max_reasonable_pixels} limit")

    diff = np.zeros((aligned_sample_mask.shape[0], aligned_sample_mask.shape[1], 3), dtype=np.uint8)
    diff[allowed_white] = (0, 80, 0)
    diff[required_white] = (0, 255, 0)
    diff[metrics["missing_required_mask"]] = (0, 0, 255)
    diff[metrics["outside_allowed_mask"]] = (255, 0, 0)

    debug_paths = {}
    if bool(inspection_cfg.get("save_debug_images", True)):
        stem = image_path.stem
        debug_paths = save_debug_outputs(stem, aligned_sample_mask, diff)

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
        "required_coverage": required_coverage,
        "outside_allowed_ratio": outside_allowed_ratio,
        "min_section_coverage": min_section_coverage,
        "section_coverages": metrics["section_coverages"],
        "sample_white_pixels": metrics["sample_white_pixels"],
        "min_required_coverage": min_required_coverage,
        "max_outside_allowed_ratio": max_outside_allowed_ratio,
        "min_section_coverage_limit": min_section_coverage_limit,
        "min_ssim": threshold_summary.get("min_ssim"),
        "max_mse": threshold_summary.get("max_mse"),
        "min_anomaly_score": threshold_summary.get("min_anomaly_score"),
        "inspection_mode": threshold_summary.get("inspection_mode", "mask_only"),
        "ssim_gate_active": bool(threshold_summary.get("ssim_gate_active", False)),
        "mse_gate_active": bool(threshold_summary.get("mse_gate_active", False)),
        "anomaly_gate_active": bool(threshold_summary.get("anomaly_gate_active", False)),
        "debug_paths": debug_paths,
        **anomaly_metrics,
    }
    return passed, details
#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

from inspection_system.app.anomaly_detection_utils import AnomalyDetector
from inspection_system.app.alignment_utils import align_sample_mask
from inspection_system.app.frame_acquisition import capture_to_temp, cleanup_temp_image
from inspection_system.app.inspection_pipeline import inspect_against_references
from inspection_system.app.morphology_utils import dilate_mask, erode_mask
from inspection_system.app.preprocessing_utils import make_binary_mask
from inspection_system.app.reference_service import (
    MIN_ANOMALY_TRAINING_SAMPLES,
    get_anomaly_model_artifact_paths,
    get_anomaly_model_metadata,
    list_anomaly_training_samples,
)
from inspection_system.app.reference_region_utils import build_reference_regions
from inspection_system.app.reference_service import list_runtime_reference_candidates, save_debug_outputs
from inspection_system.app.scoring_utils import evaluate_metrics, normalize_inspection_mode, score_sample
from inspection_system.app.section_mask_utils import compute_section_masks
from inspection_system.app.camera_interface import import_cv2_and_numpy, get_active_runtime_paths


def _optional_float(value):
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def describe_edge_gate_status(config: dict) -> tuple[str, str | None]:
    inspection_cfg = config.get("inspection", {})
    max_mean_edge_distance_px = _optional_float(inspection_cfg.get("max_mean_edge_distance_px"))
    max_section_edge_distance_px = _optional_float(inspection_cfg.get("max_section_edge_distance_px"))

    mean_status = (
        f"global<={max_mean_edge_distance_px:.2f}px" if max_mean_edge_distance_px is not None else "global off"
    )
    section_status = (
        f"section<={max_section_edge_distance_px:.2f}px"
        if max_section_edge_distance_px is not None
        else "section off"
    )
    status_line = f"Edge Gates: {mean_status} | {section_status}"

    if max_mean_edge_distance_px is None and max_section_edge_distance_px is None:
        return (
            status_line,
            "Hint: set Max Mean Edge Distance and Max Section Edge Distance to enable global and section edge drift checks.",
        )
    if max_mean_edge_distance_px is None:
        return (
            status_line,
            "Hint: set Max Mean Edge Distance to enable the global edge drift gate.",
        )
    if max_section_edge_distance_px is None:
        return (
            status_line,
            "Hint: set Max Section Edge Distance to enable the section edge drift gate.",
        )
    return status_line, None


def describe_section_width_gate_status(config: dict) -> tuple[str, str | None]:
    inspection_cfg = config.get("inspection", {})
    max_section_width_delta_ratio = _optional_float(inspection_cfg.get("max_section_width_delta_ratio"))
    if max_section_width_delta_ratio is None:
        return (
            "Width Gate: section off",
            "Hint: set Max Section Width Drift to enable per-section width drift checks.",
        )
    return (f"Width Gate: section<={max_section_width_delta_ratio:.1%}", None)


def describe_section_center_gate_status(config: dict) -> tuple[str, str | None]:
    inspection_cfg = config.get("inspection", {})
    max_section_center_offset_px = _optional_float(inspection_cfg.get("max_section_center_offset_px"))
    if max_section_center_offset_px is None:
        return (
            "Center Gate: section off",
            "Hint: set Max Section Center Offset to enable per-section center drift checks.",
        )
    return (f"Center Gate: section<={max_section_center_offset_px:.2f}px", None)


def load_anomaly_detector(active_paths: dict):
    model_path = Path(active_paths["reference_dir"]) / "anomaly_model.pkl"
    model_path = get_anomaly_model_artifact_paths(active_paths)["model"]
    if not model_path.exists():
        return None

    detector = AnomalyDetector(model_path=model_path)
    try:
        detector.load_model()
        return detector
    except Exception as exc:
        print(f"Warning: failed to load anomaly model from {model_path}: {exc}")
        return None


def get_anomaly_model_status(
    config: dict,
    anomaly_detector: Optional[AnomalyDetector],
    active_paths: Optional[dict] = None,
) -> dict:
    inspection_cfg = config.get("inspection", {})
    inspection_mode = normalize_inspection_mode(inspection_cfg.get("inspection_mode", "mask_only"))
    status = {
        "inspection_mode": inspection_mode,
        "ml_mode_selected": inspection_mode in {"mask_and_ml", "full"},
        "active_good_samples": 0,
        "pending_good_samples": 0,
        "minimum_required_samples": MIN_ANOMALY_TRAINING_SAMPLES,
        "model_path": None,
        "model_exists": False,
        "trained_sample_count": None,
        "model_stale": False,
        "ready": False,
    }

    if active_paths is None:
        status["model_exists"] = anomaly_detector is not None
        status["ready"] = anomaly_detector is not None
        return status

    artifact_paths = get_anomaly_model_artifact_paths(active_paths)
    metadata = get_anomaly_model_metadata(active_paths) or {}
    active_samples = list_anomaly_training_samples(active_paths, states=("active",))
    pending_samples = list_anomaly_training_samples(active_paths, states=("pending",))

    trained_sample_count = metadata.get("trained_sample_count")
    if trained_sample_count is not None:
        trained_sample_count = int(trained_sample_count)

    model_exists = artifact_paths["model"].exists()
    model_stale = False
    if model_exists and trained_sample_count is not None:
        model_stale = trained_sample_count != len(active_samples)

    status.update(
        {
            "active_good_samples": len(active_samples),
            "pending_good_samples": len(pending_samples),
            "model_path": str(artifact_paths["model"]),
            "model_exists": model_exists,
            "trained_sample_count": trained_sample_count,
            "model_stale": model_stale,
            "ready": (
                model_exists
                and anomaly_detector is not None
                and len(active_samples) >= MIN_ANOMALY_TRAINING_SAMPLES
                and not model_stale
            ),
        }
    )
    return status


def get_inspection_runtime_warnings(
    config: dict,
    anomaly_detector: Optional[AnomalyDetector],
    active_paths: Optional[dict] = None,
) -> list[str]:
    inspection_cfg = config.get("inspection", {})
    inspection_mode = normalize_inspection_mode(inspection_cfg.get("inspection_mode", "mask_only"))
    warnings: list[str] = []
    anomaly_status = get_anomaly_model_status(config, anomaly_detector, active_paths)

    if inspection_mode in {"mask_and_ml", "full"}:
        if active_paths is not None and anomaly_status["active_good_samples"] < anomaly_status["minimum_required_samples"]:
            warnings.append(
                "ML-backed mode is selected but there are not enough approved-good samples to train the anomaly model. "
                f"Committed good samples: {anomaly_status['active_good_samples']}/{anomaly_status['minimum_required_samples']}."
            )
        elif active_paths is not None and anomaly_status["model_stale"]:
            warnings.append(
                "ML-backed mode is selected but the anomaly model is stale for the current approved-good sample library. "
                "Press Update in training to rebuild it."
            )
        elif anomaly_detector is None:
            warnings.append(
                "ML-backed mode is selected but no trained anomaly model is available. The anomaly check will not be enforced."
            )
        if inspection_cfg.get("min_anomaly_score") in {None, ""}:
            warnings.append(
                "ML-backed mode is selected but Min Anomaly Score is not set. The anomaly gate is inactive."
            )

    return warnings


def format_operator_mode_lines(
    config: dict,
    active_paths: Optional[dict] = None,
    anomaly_detector: Optional[AnomalyDetector] = None,
) -> list[str]:
    inspection_cfg = config.get("inspection", {})
    inspection_mode = normalize_inspection_mode(inspection_cfg.get("inspection_mode", "mask_only"))
    reference_strategy = str(inspection_cfg.get("reference_strategy", "golden_only")).strip().lower() or "golden_only"
    blend_mode = str(inspection_cfg.get("blend_mode", "hard_only")).strip().lower() or "hard_only"
    tolerance_mode = str(inspection_cfg.get("tolerance_mode", "balanced")).strip().lower() or "balanced"
    lines = [
        f"Mode: {inspection_mode} | Ref: {reference_strategy}",
        f"Blend: {blend_mode} | Tol: {tolerance_mode}",
    ]
    edge_status_line, edge_hint = describe_edge_gate_status(config)
    lines.append(edge_status_line)
    if edge_hint:
        lines.append(edge_hint)
    width_status_line, width_hint = describe_section_width_gate_status(config)
    lines.append(width_status_line)
    if width_hint:
        lines.append(width_hint)
    center_status_line, center_hint = describe_section_center_gate_status(config)
    lines.append(center_status_line)
    if center_hint:
        lines.append(center_hint)

    if active_paths is not None:
        reference_count = len(list_runtime_reference_candidates(config, active_paths))
        lines[0] = f"Mode: {inspection_mode} | Ref: {reference_strategy} ({reference_count})"
        anomaly_status = get_anomaly_model_status(config, anomaly_detector, active_paths)
        if anomaly_status["ml_mode_selected"]:
            if anomaly_status["ready"]:
                lines.append(f"ML: ready ({anomaly_status['active_good_samples']} approved-good samples)")
            else:
                lines.append(
                    "ML: "
                    f"{anomaly_status['active_good_samples']}/{anomaly_status['minimum_required_samples']} approved-good samples"
                )

    return lines


def print_inspection_runtime_warnings(config: dict, anomaly_detector: Optional[AnomalyDetector]) -> list[str]:
    warnings = get_inspection_runtime_warnings(config, anomaly_detector)
    for warning in warnings:
        print(f"Warning: {warning}")
    return warnings


def run_interactive_training(config: dict) -> int:
    """Import and run interactive training mode."""
    try:
        from inspection_system.app.interactive_training import run_interactive_training as train_func
        return train_func(config)
    except ImportError as exc:
        print(f"Interactive training not available: {exc}")
        return 1


def run_production_mode(config: dict, indicator) -> int:
    """Import and run production inspection mode."""
    try:
        from inspection_system.app.production_screen import run_production_mode as production_func

        return production_func(config, indicator)
    except ImportError as exc:
        print(f"Production mode not available: {exc}")
        return 1


def print_inspection_result(passed: bool, details: dict) -> None:
    print("Inspection result:", "PASS" if passed else "FAIL")
    print(f"Inspection mode: {details.get('inspection_mode', 'mask_only')}")
    if details.get("reference_label"):
        print(
            f"Selected reference: {details.get('reference_label')}"
            f" ({details.get('reference_role', 'candidate')})"
        )
    print(f"ROI: {details['roi']}")
    print(f"Best angle correction: {details.get('best_angle_deg', 0.0):.2f} deg")
    print(f"Best shift correction: x={details.get('best_shift_x', 0)}, y={details.get('best_shift_y', 0)} px")
    print(f"Required coverage: {details['required_coverage']:.4f} (min {details['min_required_coverage']:.4f})")
    print(f"Outside allowed ratio: {details['outside_allowed_ratio']:.4f} (max {details['max_outside_allowed_ratio']:.4f})")
    print(f"Min section coverage: {details['min_section_coverage']:.4f} (min {details['min_section_coverage_limit']:.4f})")
    if details.get("worst_section_edge_distance_px") is not None:
        section_edge_gate_active = bool(details.get("section_edge_gate_active", False))
        if details.get("max_section_edge_distance_px") is not None:
            suffix = " [gate]" if section_edge_gate_active else " [info]"
            print(
                "Worst section edge distance: "
                f"{details['worst_section_edge_distance_px']:.3f}px "
                f"(max {details['max_section_edge_distance_px']:.3f}px){suffix}"
            )
        else:
            print(f"Worst section edge distance: {details['worst_section_edge_distance_px']:.3f}px")
    if details.get("worst_section_width_delta_ratio") is not None:
        section_width_gate_active = bool(details.get("section_width_gate_active", False))
        if details.get("max_section_width_delta_ratio") is not None:
            suffix = " [gate]" if section_width_gate_active else " [info]"
            print(
                "Worst section width drift: "
                f"{details['worst_section_width_delta_ratio']:.1%} "
                f"(max {details['max_section_width_delta_ratio']:.1%}){suffix}"
            )
        else:
            print(f"Worst section width drift: {details['worst_section_width_delta_ratio']:.1%}")
    if details.get("worst_section_center_offset_px") is not None:
        section_center_gate_active = bool(details.get("section_center_gate_active", False))
        if details.get("max_section_center_offset_px") is not None:
            suffix = " [gate]" if section_center_gate_active else " [info]"
            print(
                "Worst section center offset: "
                f"{details['worst_section_center_offset_px']:.3f}px "
                f"(max {details['max_section_center_offset_px']:.3f}px){suffix}"
            )
        else:
            print(f"Worst section center offset: {details['worst_section_center_offset_px']:.3f}px")
    if details.get("mean_edge_distance_px") is not None:
        edge_distance_gate_active = bool(details.get("edge_distance_gate_active", False))
        if details.get("max_mean_edge_distance_px") is not None:
            suffix = " [gate]" if edge_distance_gate_active else " [info]"
            print(
                "Mean edge distance: "
                f"{details['mean_edge_distance_px']:.3f}px "
                f"(max {details['max_mean_edge_distance_px']:.3f}px){suffix}"
            )
        else:
            print(f"Mean edge distance: {details['mean_edge_distance_px']:.3f}px")
    print(f"Sample white pixels: {details['sample_white_pixels']}")
    if details.get("section_coverages"):
        print("Section coverages:", ", ".join(f"{v:.3f}" for v in details["section_coverages"]))
    if details.get("section_edge_distances_px"):
        print("Section edge distances:", ", ".join(f"{v:.3f}" for v in details["section_edge_distances_px"]))
    if details.get("section_width_ratios"):
        print("Section width ratios:", ", ".join(f"{v:.3f}x" for v in details["section_width_ratios"]))
    if details.get("section_center_offsets_px"):
        print("Section center offsets:", ", ".join(f"{v:.3f}px" for v in details["section_center_offsets_px"]))
    if "ssim" in details:
        ssim_gate_active = bool(details.get("ssim_gate_active", False))
        if details.get("min_ssim") is not None:
            suffix = " [gate]" if ssim_gate_active else " [info]"
            print(f"SSIM: {details['ssim']:.4f} (min {details['min_ssim']:.4f}){suffix}")
        else:
            print(f"SSIM: {details['ssim']:.4f}")
    if "histogram_similarity" in details:
        print(f"Histogram similarity: {details['histogram_similarity']:.4f}")
    if "mse" in details:
        mse_gate_active = bool(details.get("mse_gate_active", False))
        if details.get("max_mse") is not None:
            suffix = " [gate]" if mse_gate_active else " [info]"
            print(f"MSE: {details['mse']:.2f} (max {details['max_mse']:.2f}){suffix}")
        else:
            print(f"MSE: {details['mse']:.2f}")
    if "anomaly_score" in details:
        anomaly_score = details.get("anomaly_score")
        anomaly_gate_active = bool(details.get("anomaly_gate_active", False))
        if anomaly_score is not None and details.get("min_anomaly_score") is not None:
            suffix = " [gate]" if anomaly_gate_active else " [info]"
            print(f"Anomaly score: {anomaly_score:.4f} (min {details['min_anomaly_score']:.4f}){suffix}")
        elif anomaly_score is not None:
            print(f"Anomaly score: {anomaly_score:.4f}")
    if details.get("debug_paths"):
        for key, path in details["debug_paths"].items():
            print(f"Debug {key}: {path}")


def run_capture_only(config: dict) -> int:
    result_code, image_path, stderr_text = capture_to_temp(config)
    if result_code != 0:
        print("Capture failed.")
        if stderr_text:
            print(stderr_text)
        cleanup_temp_image()
        return result_code

    print("Temporary capture completed.")
    cleanup_temp_image()
    return 0


def run_capture_and_inspect(config: dict, indicator) -> int:
    result_code, image_path, stderr_text = capture_to_temp(config)
    if result_code != 0:
        print("Capture failed.")
        if stderr_text:
            print(stderr_text)
        indicator.pulse_fail()
        cleanup_temp_image()
        return result_code

    try:
        active_paths = get_active_runtime_paths()
        anomaly_detector = load_anomaly_detector(active_paths)
        for warning in get_inspection_runtime_warnings(config, anomaly_detector, active_paths):
            print(f"Warning: {warning}")
        reference_candidates = list_runtime_reference_candidates(config, active_paths)
        if not reference_candidates:
            print("No active runtime references are available. Capture a golden reference first.")
            indicator.pulse_fail()
            return 1
        passed, details = inspect_against_references(
            config,
            image_path,
            reference_candidates,
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
            anomaly_detector=anomaly_detector,
        )
        print_inspection_result(passed, details)
        if passed:
            indicator.pulse_pass()
            return 0

        indicator.pulse_fail()
        return 1
    finally:
        cleanup_temp_image()

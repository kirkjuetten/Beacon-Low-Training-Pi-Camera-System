#!/usr/bin/env python3
from pathlib import Path

from inspection_system.app.anomaly_detection_utils import AnomalyDetector
from inspection_system.app.alignment_utils import align_sample_mask
from inspection_system.app.frame_acquisition import capture_to_temp, cleanup_temp_image
from inspection_system.app.inspection_pipeline import inspect_against_reference
from inspection_system.app.morphology_utils import dilate_mask, erode_mask
from inspection_system.app.preprocessing_utils import make_binary_mask
from inspection_system.app.reference_region_utils import build_reference_regions
from inspection_system.app.reference_service import save_debug_outputs
from inspection_system.app.scoring_utils import evaluate_metrics, score_sample
from inspection_system.app.section_mask_utils import compute_section_masks
from inspection_system.app.camera_interface import import_cv2_and_numpy, get_active_runtime_paths


def load_anomaly_detector(active_paths: dict):
    model_path = Path(active_paths["reference_dir"]) / "anomaly_model.pkl"
    if not model_path.exists():
        return None

    detector = AnomalyDetector(model_path=model_path)
    try:
        detector.load_model()
        return detector
    except Exception as exc:
        print(f"Warning: failed to load anomaly model from {model_path}: {exc}")
        return None


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
    print(f"ROI: {details['roi']}")
    print(f"Best angle correction: {details.get('best_angle_deg', 0.0):.2f} deg")
    print(f"Best shift correction: x={details.get('best_shift_x', 0)}, y={details.get('best_shift_y', 0)} px")
    print(f"Required coverage: {details['required_coverage']:.4f} (min {details['min_required_coverage']:.4f})")
    print(f"Outside allowed ratio: {details['outside_allowed_ratio']:.4f} (max {details['max_outside_allowed_ratio']:.4f})")
    print(f"Min section coverage: {details['min_section_coverage']:.4f} (min {details['min_section_coverage_limit']:.4f})")
    print(f"Sample white pixels: {details['sample_white_pixels']}")
    if details.get("section_coverages"):
        print("Section coverages:", ", ".join(f"{v:.3f}" for v in details["section_coverages"]))
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
        passed, details = inspect_against_reference(
            config,
            image_path,
            make_binary_mask,
            active_paths["reference_mask"],
            active_paths["reference_image"],
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

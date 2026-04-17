#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from inspection_system.app import camera_interface
from inspection_system.app.camera_interface import (
    CONFIG_FILE,
    REFERENCE_MASK,
    REFERENCE_IMAGE,
    get_active_runtime_paths,
    load_config,
    import_cv2_and_numpy,
)
from inspection_system.app.inspection_pipeline import inspect_against_references
from inspection_system.app.reference_service import list_runtime_reference_candidates, save_debug_outputs
from inspection_system.app.runtime_controller import load_anomaly_detector
from inspection_system.app.alignment_utils import align_sample_mask
from inspection_system.app.morphology_utils import dilate_mask, erode_mask
from inspection_system.app.preprocessing_utils import make_binary_mask
from inspection_system.app.reference_region_utils import build_reference_regions
from inspection_system.app.scoring_utils import evaluate_metrics, score_sample
from inspection_system.app.section_mask_utils import compute_section_masks
from inspection_system.app.result_status import CONFIG_ERROR, FAIL, INVALID_CAPTURE, PASS

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _get_reference_mask_path() -> Path:
    module_mask = getattr(camera_interface, "REFERENCE_MASK", None)
    local_mask = globals().get("REFERENCE_MASK", None)

    if module_mask is not None and not Path(module_mask).exists():
        return Path(module_mask)

    if local_mask is not None and not Path(local_mask).exists():
        return Path(local_mask)

    return Path(module_mask if module_mask is not None else local_mask)


def _get_reference_image_path() -> Path:
    active_paths = get_active_runtime_paths()
    runtime_image = active_paths.get("reference_image")
    if runtime_image is not None:
        return Path(runtime_image)
    return Path(REFERENCE_IMAGE)


def _round_optional_float(value):
    if value is None:
        return None
    return round(float(value), 6)


def classify_invalid_capture(config: dict, image_path: Path, active_paths: dict | None = None) -> str | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return "OpenCV is not installed."

    if not image_path.exists():
        return f"Image does not exist: {image_path}"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return f"Image could not be read: {image_path}"

    roi_cfg = config.get("inspection", {}).get("roi", {})
    x = int(roi_cfg.get("x", 0))
    y = int(roi_cfg.get("y", 0))
    w = int(roi_cfg.get("width", 0))
    h = int(roi_cfg.get("height", 0))

    if w > 0 and h > 0:
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            return "Configured ROI is outside image bounds."

    runtime_paths = active_paths or get_active_runtime_paths()
    if not list_runtime_reference_candidates(config, runtime_paths):
        reference_mask = _get_reference_mask_path()
        return f"Reference mask is missing: {reference_mask}"

    return None


def inspect_file(config: dict, image_path: Path, active_paths: dict | None = None) -> dict:
    invalid_reason = classify_invalid_capture(config, image_path, active_paths=active_paths)
    if invalid_reason is not None:
        return {
            "image": str(image_path),
            "status": INVALID_CAPTURE,
            "reason": invalid_reason,
        }

    runtime_paths = active_paths or get_active_runtime_paths()
    reference_candidates = list_runtime_reference_candidates(config, runtime_paths)
    anomaly_detector = load_anomaly_detector(runtime_paths)

    try:
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
    except FileNotFoundError as exc:
        return {
            "image": str(image_path),
            "status": CONFIG_ERROR,
            "reason": str(exc),
        }
    except ValueError as exc:
        return {
            "image": str(image_path),
            "status": INVALID_CAPTURE,
            "reason": str(exc),
        }
    except Exception as exc:
        return {
            "image": str(image_path),
            "status": CONFIG_ERROR,
            "reason": str(exc),
        }

    return {
        "image": str(image_path),
        "status": PASS if passed else FAIL,
        "required_coverage": round(float(details.get("required_coverage", 0.0)), 6),
        "outside_allowed_ratio": round(float(details.get("outside_allowed_ratio", 0.0)), 6),
        "min_section_coverage": round(float(details.get("min_section_coverage", 0.0)), 6),
        "min_required_coverage": _round_optional_float(details.get("min_required_coverage")),
        "max_outside_allowed_ratio": _round_optional_float(details.get("max_outside_allowed_ratio")),
        "min_section_coverage_limit": _round_optional_float(details.get("min_section_coverage_limit")),
        "effective_min_required_coverage": _round_optional_float(details.get("effective_min_required_coverage")),
        "effective_max_outside_allowed_ratio": _round_optional_float(details.get("effective_max_outside_allowed_ratio")),
        "effective_min_section_coverage": _round_optional_float(details.get("effective_min_section_coverage")),
        "sample_white_pixels": int(details.get("sample_white_pixels", 0)),
        "best_angle_deg": round(float(details.get("best_angle_deg", 0.0)), 6),
        "best_shift_x": int(details.get("best_shift_x", 0)),
        "best_shift_y": int(details.get("best_shift_y", 0)),
        "mean_edge_distance_px": _round_optional_float(details.get("mean_edge_distance_px")),
        "max_mean_edge_distance_px": _round_optional_float(details.get("max_mean_edge_distance_px")),
        "effective_max_mean_edge_distance_px": _round_optional_float(details.get("effective_max_mean_edge_distance_px")),
        "worst_section_edge_distance_px": _round_optional_float(details.get("worst_section_edge_distance_px")),
        "max_section_edge_distance_px": _round_optional_float(details.get("max_section_edge_distance_px")),
        "effective_max_section_edge_distance_px": _round_optional_float(details.get("effective_max_section_edge_distance_px")),
        "worst_section_width_delta_ratio": _round_optional_float(details.get("worst_section_width_delta_ratio")),
        "max_section_width_delta_ratio": _round_optional_float(details.get("max_section_width_delta_ratio")),
        "effective_max_section_width_delta_ratio": _round_optional_float(details.get("effective_max_section_width_delta_ratio")),
        "worst_section_center_offset_px": _round_optional_float(details.get("worst_section_center_offset_px")),
        "max_section_center_offset_px": _round_optional_float(details.get("max_section_center_offset_px")),
        "effective_max_section_center_offset_px": _round_optional_float(details.get("effective_max_section_center_offset_px")),
        "ssim": _round_optional_float(details.get("ssim")),
        "min_ssim": _round_optional_float(details.get("min_ssim")),
        "effective_min_ssim": _round_optional_float(details.get("effective_min_ssim")),
        "mse": _round_optional_float(details.get("mse")),
        "max_mse": _round_optional_float(details.get("max_mse")),
        "effective_max_mse": _round_optional_float(details.get("effective_max_mse")),
        "anomaly_score": _round_optional_float(details.get("anomaly_score")),
        "min_anomaly_score": _round_optional_float(details.get("min_anomaly_score")),
        "effective_min_anomaly_score": _round_optional_float(details.get("effective_min_anomaly_score")),
        "histogram_similarity": _round_optional_float(details.get("histogram_similarity")),
        "inspection_mode": details.get("inspection_mode", config.get("inspection", {}).get("inspection_mode", "mask_only")),
        "edge_distance_gate_active": bool(details.get("edge_distance_gate_active", False)),
        "section_edge_gate_active": bool(details.get("section_edge_gate_active", False)),
        "section_width_gate_active": bool(details.get("section_width_gate_active", False)),
        "section_center_gate_active": bool(details.get("section_center_gate_active", False)),
        "ssim_gate_active": bool(details.get("ssim_gate_active", False)),
        "mse_gate_active": bool(details.get("mse_gate_active", False)),
        "anomaly_gate_active": bool(details.get("anomaly_gate_active", False)),
    }


def inspect_folder(config: dict, folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        print(json.dumps({"status": CONFIG_ERROR, "reason": f"Folder not found: {folder}"}))
        return 2

    image_paths = sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )

    if not image_paths:
        print(json.dumps({"status": CONFIG_ERROR, "reason": f"No images found in: {folder}"}))
        return 2

    exit_code = 0
    for image_path in image_paths:
        result = inspect_file(config, image_path)
        print(json.dumps(result, sort_keys=True))
        if result["status"] in {FAIL, INVALID_CAPTURE, CONFIG_ERROR}:
            exit_code = 1

    return exit_code


def print_usage() -> None:
    print("Usage:")
    print("  python3 inspection_system/app/replay_inspection.py inspect-file <image_path>")
    print("  python3 inspection_system/app/replay_inspection.py inspect-folder <folder_path>")
    print(f"  Config file expected at: {CONFIG_FILE}")
    print(f"  Golden reference mask expected at: {_get_reference_mask_path()}")
    print(f"  Golden reference image expected at: {_get_reference_image_path()}")


def main() -> int:
    if len(sys.argv) != 3:
        print_usage()
        return 2

    mode = sys.argv[1]
    target = Path(sys.argv[2])
    config = load_config()

    if mode == "inspect-file":
        result = inspect_file(config, target)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == PASS else 1

    if mode == "inspect-folder":
        return inspect_folder(config, target)

    print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

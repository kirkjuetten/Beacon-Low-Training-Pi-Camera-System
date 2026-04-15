#!/usr/bin/env python3
import json
import time
from pathlib import Path

from inspection_system.app.camera_interface import get_active_runtime_paths, import_cv2_and_numpy
from inspection_system.app.frame_acquisition import capture_to_temp, cleanup_temp_image
from inspection_system.app.morphology_utils import dilate_mask, erode_mask
from inspection_system.app.preprocessing_utils import make_binary_mask


def save_debug_outputs(stem: str, aligned_sample_mask, diff_image) -> dict:
    cv2, _ = import_cv2_and_numpy()
    active_paths = get_active_runtime_paths()
    debug_mask_path = active_paths["reference_dir"] / f"{stem}_mask.png"
    debug_diff_path = active_paths["reference_dir"] / f"{stem}_diff.png"
    cv2.imwrite(str(debug_mask_path), aligned_sample_mask)
    cv2.imwrite(str(debug_diff_path), diff_image)
    return {
        "mask": str(debug_mask_path),
        "diff": str(debug_diff_path),
    }


def bake_reference_mask(image_path: Path, config: dict) -> tuple[object, object, int, str | None]:
    """
    Pure reference-baking function: process image into reference assets.
    
    Returns:
        (roi_image, mask, feature_pixels, error_message)
        If error_message is None, baking succeeded.
    """
    try:
        cv2, np = import_cv2_and_numpy()
        inspection_cfg = config.get("inspection", {})
        roi_image, _, mask, _, _, _ = make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy)
        reference_erode_iterations = int(inspection_cfg.get("reference_erode_iterations", 1))
        reference_dilate_iterations = int(inspection_cfg.get("reference_dilate_iterations", 1))
        mask = erode_mask(mask, reference_erode_iterations, cv2, np)
        mask = dilate_mask(mask, reference_dilate_iterations, cv2, np)

        feature_pixels = int((mask > 0).sum())
        min_feature_pixels = int(inspection_cfg.get("min_feature_pixels", inspection_cfg.get("min_white_pixels", 100)))
        if feature_pixels < min_feature_pixels:
            error_msg = f"Too few feature pixels ({feature_pixels}). Adjust ROI or threshold."
            return None, None, 0, error_msg

        return roi_image, mask, feature_pixels, None
    except Exception as exc:
        return None, None, 0, f"Reference baking error: {exc}"


def save_reference_metadata(config: dict) -> None:
    """Save reference creation settings as fingerprint for later validation."""
    try:
        inspection_cfg = config.get("inspection", {})
        metadata = {
            "created_at": time.time(),
            "roi": {
                "x1": int(inspection_cfg.get("roi_x1", 0)),
                "y1": int(inspection_cfg.get("roi_y1", 0)),
                "x2": int(inspection_cfg.get("roi_x2", 640)),
                "y2": int(inspection_cfg.get("roi_y2", 480)),
            },
            "threshold": {
                "type": str(inspection_cfg.get("mask_threshold_type", "fixed")).lower(),
                "value": float(inspection_cfg.get("mask_threshold_value", 150.0)),
                "blur_kernel": int(inspection_cfg.get("mask_blur_kernel", 5)),
            },
            "morphology": {
                "reference_erode_iterations": int(inspection_cfg.get("reference_erode_iterations", 1)),
                "reference_dilate_iterations": int(inspection_cfg.get("reference_dilate_iterations", 1)),
            },
        }
        active_paths = get_active_runtime_paths()
        meta_path = active_paths["reference_dir"] / "ref_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as exc:
        # Don't crash on metadata save; just warn
        import sys
        print(f"Warning: Could not save reference metadata: {exc}", file=sys.stderr)


def load_reference_metadata() -> dict | None:
    """Load reference creation settings; returns None if metadata doesn't exist."""
    try:
        active_paths = get_active_runtime_paths()
        meta_path = active_paths["reference_dir"] / "ref_meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def check_reference_settings_match(config: dict) -> tuple[bool, str | None]:
    """Compare current config against reference metadata. Returns (match, warning_msg)."""
    meta = load_reference_metadata()
    if meta is None:
        return True, None  # No metadata to compare
    
    inspection_cfg = config.get("inspection", {})
    
    # Check ROI
    roi_meta = meta.get("roi", {})
    current_roi = (
        int(inspection_cfg.get("roi_x1", 0)),
        int(inspection_cfg.get("roi_y1", 0)),
        int(inspection_cfg.get("roi_x2", 640)),
        int(inspection_cfg.get("roi_y2", 480)),
    )
    meta_roi = (roi_meta.get("x1", 0), roi_meta.get("y1", 0), roi_meta.get("x2", 640), roi_meta.get("y2", 480))
    if current_roi != meta_roi:
        return False, f"ROI mismatch: reference was {meta_roi}; current is {current_roi}"
    
    # Check threshold
    thresh_meta = meta.get("threshold", {})
    current_thresh_type = str(inspection_cfg.get("mask_threshold_type", "fixed")).lower()
    meta_thresh_type = str(thresh_meta.get("type", "fixed")).lower()
    if current_thresh_type != meta_thresh_type:
        return False, f"Threshold type mismatch: reference was {meta_thresh_type}; current is {current_thresh_type}"
    
    current_thresh_val = float(inspection_cfg.get("mask_threshold_value", 150.0))
    meta_thresh_val = float(thresh_meta.get("value", 150.0))
    if abs(current_thresh_val - meta_thresh_val) > 0.1:
        return False, f"Threshold value mismatch: reference was {meta_thresh_val}; current is {current_thresh_val}"
    
    # Check morphology
    morph_meta = meta.get("morphology", {})
    current_ref_erode = int(inspection_cfg.get("reference_erode_iterations", 1))
    meta_ref_erode = int(morph_meta.get("reference_erode_iterations", 1))
    if current_ref_erode != meta_ref_erode:
        return False, f"Reference erode iterations mismatch: {meta_ref_erode} -> {current_ref_erode}"
    
    current_ref_dilate = int(inspection_cfg.get("reference_dilate_iterations", 1))
    meta_ref_dilate = int(morph_meta.get("reference_dilate_iterations", 1))
    if current_ref_dilate != meta_ref_dilate:
        return False, f"Reference dilate iterations mismatch: {meta_ref_dilate} -> {current_ref_dilate}"
    
    return True, None


def set_reference(config: dict) -> int:
    result_code, image_path, stderr_text = capture_to_temp(config)
    if result_code != 0:
        print("Reference capture failed.")
        if stderr_text:
            print(stderr_text)
        cleanup_temp_image()
        return result_code

    try:
        roi_image, mask, feature_pixels, error_msg = bake_reference_mask(image_path, config)
        if error_msg:
            print(error_msg)
            return 3

        active_paths = get_active_runtime_paths()
        ref_mask_path = active_paths["reference_mask"]
        ref_image_path = active_paths["reference_image"]
        ref_mask_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2, _ = import_cv2_and_numpy()
        cv2.imwrite(str(ref_mask_path), mask)
        cv2.imwrite(str(ref_image_path), roi_image)
        print(f"Saved reference mask: {ref_mask_path}")
        print(f"Saved reference image: {ref_image_path}")
        print(f"Reference feature pixels: {feature_pixels}")
        try:
            save_reference_metadata(config)
        except Exception as exc:
            print(f"Warning: could not save reference metadata: {exc}")
        return 0
    finally:
        cleanup_temp_image()

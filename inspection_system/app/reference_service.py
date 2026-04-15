#!/usr/bin/env python3
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
        return 0
    finally:
        cleanup_temp_image()

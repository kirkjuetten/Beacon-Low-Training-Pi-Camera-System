#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

from inspection_system.app.anomaly_detection_utils import AnomalyDetector
from inspection_system.app.camera_interface import get_active_runtime_paths, import_cv2_and_numpy
from inspection_system.app.frame_acquisition import capture_to_temp, cleanup_temp_image
from inspection_system.app.reference_library import (
    ANOMALY_MODEL_FILENAME,
    ANOMALY_MODEL_METADATA_FILENAME,
    ANOMALY_TRAINING_LIBRARY_DIRNAME,
    MIN_ANOMALY_TRAINING_SAMPLES,
    REFERENCE_VARIANTS_ACTIVE_DIRNAME,
    REFERENCE_VARIANTS_DIRNAME,
    REFERENCE_VARIANTS_PENDING_DIRNAME,
    activate_anomaly_training_sample,
    activate_reference_candidate,
    bake_reference_mask,
    clear_anomaly_training_artifacts,
    clear_reference_variants,
    discard_anomaly_training_sample,
    discard_reference_candidate,
    get_anomaly_model_artifact_paths,
    get_anomaly_model_metadata,
    get_anomaly_training_directories,
    get_reference_variant_directories,
    list_anomaly_training_samples,
    list_runtime_reference_candidates,
    save_debug_outputs,
    stage_anomaly_training_sample_from_image,
    stage_reference_candidate_from_image,
    train_anomaly_model_from_samples_impl,
)
from inspection_system.app.reference_metadata import (
    _build_reference_metadata,
    _extract_meta_roi_tuple,
    _extract_roi_tuple,
    _load_reference_metadata_from_path,
    _write_reference_metadata,
    check_reference_settings_match_impl,
    load_reference_metadata,
    save_reference_metadata,
)
from inspection_system.app.reference_registration import (
    build_registration_commissioning_summary,
    registration_baseline_matches_config,
)


def check_reference_settings_match(config: dict) -> tuple[bool, str | None]:
    return check_reference_settings_match_impl(config, metadata_loader=load_reference_metadata)


def train_anomaly_model_from_samples(
    config: dict,
    active_paths: Optional[dict] = None,
    *,
    minimum_samples: int = MIN_ANOMALY_TRAINING_SAMPLES,
) -> dict:
    return train_anomaly_model_from_samples_impl(
        config,
        active_paths,
        minimum_samples=minimum_samples,
        detector_cls=AnomalyDetector,
        build_reference_metadata_fn=_build_reference_metadata,
    )


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
        clear_reference_variants(active_paths)
        clear_anomaly_training_artifacts(active_paths)
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

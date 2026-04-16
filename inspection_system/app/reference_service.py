#!/usr/bin/env python3
import json
import shutil
import time
from pathlib import Path
from typing import Optional

from inspection_system.app.anomaly_detection_utils import AnomalyDetector, extract_features
from inspection_system.app.camera_interface import get_active_runtime_paths, import_cv2_and_numpy
from inspection_system.app.frame_acquisition import capture_to_temp, cleanup_temp_image
from inspection_system.app.morphology_utils import dilate_mask, erode_mask
from inspection_system.app.preprocessing_utils import make_binary_mask


REFERENCE_VARIANTS_DIRNAME = "reference_variants"
REFERENCE_VARIANTS_ACTIVE_DIRNAME = "active"
REFERENCE_VARIANTS_PENDING_DIRNAME = "pending"
ANOMALY_TRAINING_LIBRARY_DIRNAME = "anomaly_good_library"
ANOMALY_MODEL_FILENAME = "anomaly_model.pkl"
ANOMALY_MODEL_METADATA_FILENAME = "anomaly_model_meta.json"
MIN_ANOMALY_TRAINING_SAMPLES = 8


def get_reference_variant_directories(active_paths: Optional[dict] = None) -> dict[str, Path]:
    runtime_paths = active_paths or get_active_runtime_paths()
    root = runtime_paths["reference_dir"] / REFERENCE_VARIANTS_DIRNAME
    return {
        "root": root,
        "active": root / REFERENCE_VARIANTS_ACTIVE_DIRNAME,
        "pending": root / REFERENCE_VARIANTS_PENDING_DIRNAME,
    }


def get_anomaly_training_directories(active_paths: Optional[dict] = None) -> dict[str, Path]:
    runtime_paths = active_paths or get_active_runtime_paths()
    root = runtime_paths["reference_dir"] / ANOMALY_TRAINING_LIBRARY_DIRNAME
    return {
        "root": root,
        "active": root / REFERENCE_VARIANTS_ACTIVE_DIRNAME,
        "pending": root / REFERENCE_VARIANTS_PENDING_DIRNAME,
    }


def get_anomaly_model_artifact_paths(active_paths: Optional[dict] = None) -> dict[str, Path]:
    runtime_paths = active_paths or get_active_runtime_paths()
    return {
        "model": runtime_paths["reference_dir"] / ANOMALY_MODEL_FILENAME,
        "metadata": runtime_paths["reference_dir"] / ANOMALY_MODEL_METADATA_FILENAME,
    }


def _sanitize_reference_id(value: str) -> str:
    safe_chars = []
    for char in str(value or ""):
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    normalized = "".join(safe_chars).strip("_")
    return normalized or f"ref_{int(time.time() * 1000)}"


def _load_reference_metadata_from_path(metadata_path: Path) -> dict | None:
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _build_reference_metadata(
    config: dict,
    *,
    reference_role: str = "golden",
    extra_context: Optional[dict] = None,
) -> dict:
    inspection_cfg = config.get("inspection", {})
    alignment_cfg = config.get("alignment", {})
    roi_x, roi_y, roi_width, roi_height = _extract_roi_tuple(inspection_cfg)
    metadata = {
        "created_at": time.time(),
        "roi": {
            "x": roi_x,
            "y": roi_y,
            "width": roi_width,
            "height": roi_height,
        },
        "threshold": {
            "type": str(inspection_cfg.get("threshold_mode", "fixed")).lower(),
            "value": float(inspection_cfg.get("threshold_value", 150.0)),
            "blur_kernel": int(inspection_cfg.get("blur_kernel", 5)),
        },
        "morphology": {
            "reference_erode_iterations": int(inspection_cfg.get("reference_erode_iterations", 1)),
            "reference_dilate_iterations": int(inspection_cfg.get("reference_dilate_iterations", 1)),
        },
        "inspection_context": {
            "inspection_mode": str(inspection_cfg.get("inspection_mode", "mask_only")).lower(),
            "reference_strategy": str(inspection_cfg.get("reference_strategy", "golden_only")).lower(),
            "blend_mode": str(inspection_cfg.get("blend_mode", "hard_only")).lower(),
            "tolerance_mode": str(inspection_cfg.get("tolerance_mode", "balanced")).lower(),
        },
        "alignment": {
            "tolerance_profile": str(alignment_cfg.get("tolerance_profile", "balanced")).lower(),
        },
        "reference_asset": {
            "role": str(reference_role).lower(),
        },
    }
    if extra_context:
        metadata["reference_asset"].update(extra_context)
    return metadata


def _write_reference_metadata(metadata: dict, metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def _extract_roi_tuple(inspection_cfg: dict) -> tuple[int, int, int, int]:
    roi_cfg = inspection_cfg.get("roi")
    if isinstance(roi_cfg, dict):
        x = int(roi_cfg.get("x", 0))
        y = int(roi_cfg.get("y", 0))
        width = int(roi_cfg.get("width", 640) or 640)
        height = int(roi_cfg.get("height", 480) or 480)
        return x, y, width, height

    x1 = int(inspection_cfg.get("roi_x1", 0))
    y1 = int(inspection_cfg.get("roi_y1", 0))
    x2 = int(inspection_cfg.get("roi_x2", 640))
    y2 = int(inspection_cfg.get("roi_y2", 480))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _extract_meta_roi_tuple(meta: dict) -> tuple[int, int, int, int]:
    roi_meta = meta.get("roi", {})
    if {"x", "y", "width", "height"}.issubset(roi_meta):
        return (
            int(roi_meta.get("x", 0)),
            int(roi_meta.get("y", 0)),
            int(roi_meta.get("width", 640)),
            int(roi_meta.get("height", 480)),
        )
    x1 = int(roi_meta.get("x1", 0))
    y1 = int(roi_meta.get("y1", 0))
    x2 = int(roi_meta.get("x2", 640))
    y2 = int(roi_meta.get("y2", 480))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _build_anomaly_training_assets(image_path: Path, config: dict):
    inspection_cfg = config.get("inspection", {})
    try:
        roi_image, _gray, sample_mask, _roi, cv2, np = make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy)
        sample_erode_iterations = int(inspection_cfg.get("sample_erode_iterations", 1))
        sample_dilate_iterations = int(inspection_cfg.get("sample_dilate_iterations", 1))
        sample_mask = erode_mask(sample_mask, sample_erode_iterations, cv2, np)
        sample_mask = dilate_mask(sample_mask, sample_dilate_iterations, cv2, np)

        feature_pixels = int((sample_mask > 0).sum())
        min_feature_pixels = int(inspection_cfg.get("min_feature_pixels", inspection_cfg.get("min_white_pixels", 100)))
        if feature_pixels < min_feature_pixels:
            return None, None, None, 0, f"Too few feature pixels ({feature_pixels}) for anomaly training sample."

        features = extract_features(roi_image, sample_mask.astype(np.uint8))
        return roi_image, sample_mask, features, feature_pixels, None
    except Exception as exc:
        return None, None, None, 0, f"Anomaly training sample error: {exc}"


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


def list_runtime_reference_candidates(config: dict, active_paths: Optional[dict] = None) -> list[dict]:
    runtime_paths = active_paths or get_active_runtime_paths()
    inspection_cfg = config.get("inspection", {}) if isinstance(config, dict) else {}
    reference_strategy = str(inspection_cfg.get("reference_strategy", "golden_only")).strip().lower()
    candidates: list[dict] = []

    golden_mask = runtime_paths["reference_mask"]
    golden_image = runtime_paths["reference_image"]
    if golden_mask.exists() and golden_image.exists():
        metadata_path = runtime_paths["reference_dir"] / "ref_meta.json"
        golden_metadata = _load_reference_metadata_from_path(metadata_path) or {}
        golden_asset = golden_metadata.get("reference_asset", {})
        candidates.append(
            {
                "reference_id": str(golden_asset.get("reference_id", "golden")),
                "label": str(golden_asset.get("label", "Golden Reference")),
                "role": "golden",
                "reference_mask_path": golden_mask,
                "reference_image_path": golden_image,
                "metadata_path": metadata_path,
                "metadata": golden_metadata,
            }
        )

    variant_dirs = get_reference_variant_directories(runtime_paths)
    active_dir = variant_dirs["active"]
    variant_candidates: list[dict] = []
    if active_dir.exists():
        for candidate_dir in sorted(path for path in active_dir.iterdir() if path.is_dir()):
            mask_path = candidate_dir / "reference_mask.png"
            image_path = candidate_dir / "reference_image.png"
            if not mask_path.exists() or not image_path.exists():
                continue
            metadata_path = candidate_dir / "ref_meta.json"
            metadata = _load_reference_metadata_from_path(metadata_path) or {}
            asset = metadata.get("reference_asset", {})
            variant_candidates.append(
                {
                    "reference_id": str(asset.get("reference_id", candidate_dir.name)),
                    "label": str(asset.get("label", candidate_dir.name)),
                    "role": str(asset.get("role", "candidate")),
                    "reference_mask_path": mask_path,
                    "reference_image_path": image_path,
                    "metadata_path": metadata_path,
                    "metadata": metadata,
                }
            )

    if reference_strategy == "golden_only":
        return candidates or variant_candidates
    if reference_strategy == "multi_good_experimental" and variant_candidates:
        return variant_candidates
    return candidates + variant_candidates


def clear_reference_variants(active_paths: Optional[dict] = None) -> None:
    variant_dirs = get_reference_variant_directories(active_paths)
    if variant_dirs["root"].exists():
        shutil.rmtree(variant_dirs["root"])


def clear_anomaly_training_artifacts(active_paths: Optional[dict] = None) -> None:
    training_dirs = get_anomaly_training_directories(active_paths)
    if training_dirs["root"].exists():
        shutil.rmtree(training_dirs["root"])

    artifact_paths = get_anomaly_model_artifact_paths(active_paths)
    for artifact_path in artifact_paths.values():
        if artifact_path.exists():
            artifact_path.unlink()


def stage_reference_candidate_from_image(
    config: dict,
    image_path: Path,
    *,
    label: Optional[str] = None,
    source_record_id: Optional[str] = None,
) -> tuple[bool, dict | str]:
    roi_image, mask, feature_pixels, error_msg = bake_reference_mask(image_path, config)
    if error_msg:
        return False, error_msg

    active_paths = get_active_runtime_paths()
    variant_dirs = get_reference_variant_directories(active_paths)
    variant_dirs["pending"].mkdir(parents=True, exist_ok=True)

    reference_id = _sanitize_reference_id(source_record_id or f"candidate_{int(time.time() * 1000)}")
    candidate_dir = variant_dirs["pending"] / reference_id
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    mask_path = candidate_dir / "reference_mask.png"
    image_output_path = candidate_dir / "reference_image.png"
    metadata_path = candidate_dir / "ref_meta.json"
    display_label = label or f"Good Ref {reference_id[-6:]}"

    try:
        cv2, _ = import_cv2_and_numpy()
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(image_output_path), roi_image)
        metadata = _build_reference_metadata(
            config,
            reference_role="candidate",
            extra_context={
                "reference_id": reference_id,
                "label": display_label,
                "source": "training_approve",
                "state": "pending",
                "source_record_id": source_record_id,
            },
        )
        _write_reference_metadata(metadata, metadata_path)
    except Exception as exc:
        shutil.rmtree(candidate_dir, ignore_errors=True)
        return False, f"Reference candidate save error: {exc}"

    return True, {
        "reference_id": reference_id,
        "label": display_label,
        "reference_dir": str(candidate_dir),
        "reference_mask_path": str(mask_path),
        "reference_image_path": str(image_output_path),
        "metadata_path": str(metadata_path),
        "feature_pixels": feature_pixels,
        "state": "pending",
    }


def stage_anomaly_training_sample_from_image(
    config: dict,
    image_path: Path,
    *,
    label: Optional[str] = None,
    source_record_id: Optional[str] = None,
) -> tuple[bool, dict | str]:
    roi_image, sample_mask, features, feature_pixels, error_msg = _build_anomaly_training_assets(image_path, config)
    if error_msg:
        return False, error_msg

    active_paths = get_active_runtime_paths()
    training_dirs = get_anomaly_training_directories(active_paths)
    training_dirs["pending"].mkdir(parents=True, exist_ok=True)

    sample_id = _sanitize_reference_id(source_record_id or f"anomaly_good_{int(time.time() * 1000)}")
    sample_dir = training_dirs["pending"] / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_image_path = sample_dir / "sample_image.png"
    sample_mask_path = sample_dir / "sample_mask.png"
    metadata_path = sample_dir / "sample_meta.json"
    display_label = label or f"Approved Good {sample_id[-6:]}"

    try:
        cv2, _ = import_cv2_and_numpy()
        cv2.imwrite(str(sample_image_path), roi_image)
        cv2.imwrite(str(sample_mask_path), sample_mask)
        metadata = {
            "created_at": time.time(),
            "feature_pixels": feature_pixels,
            "feature_length": int(len(features)),
            "features": [float(value) for value in features.tolist()],
            "inspection_context": _build_reference_metadata(config).get("inspection_context", {}),
            "sample_asset": {
                "sample_id": sample_id,
                "label": display_label,
                "source": "training_good_sample",
                "state": "pending",
                "source_record_id": source_record_id,
            },
        }
        _write_reference_metadata(metadata, metadata_path)
    except Exception as exc:
        shutil.rmtree(sample_dir, ignore_errors=True)
        return False, f"Anomaly training sample save error: {exc}"

    return True, {
        "sample_id": sample_id,
        "label": display_label,
        "sample_dir": str(sample_dir),
        "sample_image_path": str(sample_image_path),
        "sample_mask_path": str(sample_mask_path),
        "metadata_path": str(metadata_path),
        "feature_pixels": feature_pixels,
        "feature_length": int(len(features)),
        "state": "pending",
    }


def _set_reference_variant_state(candidate_dir: Path, state: str) -> None:
    metadata_path = candidate_dir / "ref_meta.json"
    metadata = _load_reference_metadata_from_path(metadata_path) or {"reference_asset": {}}
    metadata.setdefault("reference_asset", {})["state"] = state
    _write_reference_metadata(metadata, metadata_path)


def _set_anomaly_training_sample_state(sample_dir: Path, state: str) -> None:
    metadata_path = sample_dir / "sample_meta.json"
    metadata = _load_reference_metadata_from_path(metadata_path) or {"sample_asset": {}}
    metadata.setdefault("sample_asset", {})["state"] = state
    _write_reference_metadata(metadata, metadata_path)


def activate_reference_candidate(reference_id: str, active_paths: Optional[dict] = None) -> bool:
    variant_dirs = get_reference_variant_directories(active_paths)
    pending_dir = variant_dirs["pending"] / _sanitize_reference_id(reference_id)
    active_dir = variant_dirs["active"] / _sanitize_reference_id(reference_id)
    if not pending_dir.exists():
        return False
    active_dir.parent.mkdir(parents=True, exist_ok=True)
    if active_dir.exists():
        shutil.rmtree(active_dir)
    shutil.move(str(pending_dir), str(active_dir))
    _set_reference_variant_state(active_dir, "active")
    return True


def activate_anomaly_training_sample(sample_id: str, active_paths: Optional[dict] = None) -> bool:
    training_dirs = get_anomaly_training_directories(active_paths)
    pending_dir = training_dirs["pending"] / _sanitize_reference_id(sample_id)
    active_dir = training_dirs["active"] / _sanitize_reference_id(sample_id)
    if not pending_dir.exists():
        return False
    active_dir.parent.mkdir(parents=True, exist_ok=True)
    if active_dir.exists():
        shutil.rmtree(active_dir)
    shutil.move(str(pending_dir), str(active_dir))
    _set_anomaly_training_sample_state(active_dir, "active")
    return True


def discard_reference_candidate(reference_id: str, active_paths: Optional[dict] = None, *, state: str = "pending") -> bool:
    variant_dirs = get_reference_variant_directories(active_paths)
    base_dir = variant_dirs["pending"] if state == "pending" else variant_dirs["active"]
    candidate_dir = base_dir / _sanitize_reference_id(reference_id)
    if not candidate_dir.exists():
        return False
    shutil.rmtree(candidate_dir)
    return True


def discard_anomaly_training_sample(sample_id: str, active_paths: Optional[dict] = None, *, state: str = "pending") -> bool:
    training_dirs = get_anomaly_training_directories(active_paths)
    base_dir = training_dirs["pending"] if state == "pending" else training_dirs["active"]
    sample_dir = base_dir / _sanitize_reference_id(sample_id)
    if not sample_dir.exists():
        return False
    shutil.rmtree(sample_dir)
    return True


def list_anomaly_training_samples(
    active_paths: Optional[dict] = None,
    *,
    states: tuple[str, ...] = ("active",),
) -> list[dict]:
    training_dirs = get_anomaly_training_directories(active_paths)
    entries: list[dict] = []
    for state in states:
        base_dir = training_dirs.get(state)
        if base_dir is None or not base_dir.exists():
            continue
        for sample_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
            metadata_path = sample_dir / "sample_meta.json"
            metadata = _load_reference_metadata_from_path(metadata_path) or {}
            sample_asset = metadata.get("sample_asset", {})
            entries.append(
                {
                    "sample_id": str(sample_asset.get("sample_id", sample_dir.name)),
                    "label": str(sample_asset.get("label", sample_dir.name)),
                    "state": state,
                    "sample_dir": sample_dir,
                    "sample_image_path": sample_dir / "sample_image.png",
                    "sample_mask_path": sample_dir / "sample_mask.png",
                    "metadata_path": metadata_path,
                    "metadata": metadata,
                    "features": metadata.get("features", []),
                }
            )
    return entries


def get_anomaly_model_metadata(active_paths: Optional[dict] = None) -> dict | None:
    artifact_paths = get_anomaly_model_artifact_paths(active_paths)
    return _load_reference_metadata_from_path(artifact_paths["metadata"])


def train_anomaly_model_from_samples(
    config: dict,
    active_paths: Optional[dict] = None,
    *,
    minimum_samples: int = MIN_ANOMALY_TRAINING_SAMPLES,
) -> dict:
    runtime_paths = active_paths or get_active_runtime_paths()
    sample_entries = list_anomaly_training_samples(runtime_paths, states=("active",))
    feature_rows = []
    sample_ids: list[str] = []
    for entry in sample_entries:
        features = entry.get("features") or []
        if not features:
            continue
        feature_rows.append(features)
        sample_ids.append(entry["sample_id"])

    artifact_paths = get_anomaly_model_artifact_paths(runtime_paths)
    trained_sample_count = len(feature_rows)
    result = {
        "rebuilt": False,
        "trained_sample_count": trained_sample_count,
        "minimum_required": int(minimum_samples),
        "model_path": str(artifact_paths["model"]),
        "metadata_path": str(artifact_paths["metadata"]),
        "reason": None,
    }

    if trained_sample_count < minimum_samples:
        for artifact_path in artifact_paths.values():
            if artifact_path.exists():
                artifact_path.unlink()
        result["reason"] = (
            f"Need at least {minimum_samples} approved-good samples to train the anomaly model; "
            f"currently have {trained_sample_count}."
        )
        return result

    detector = AnomalyDetector(model_path=artifact_paths["model"])
    try:
        detector.train(feature_rows)
        detector.save_model()
        metadata = {
            "trained_at": time.time(),
            "trained_sample_count": trained_sample_count,
            "minimum_required": int(minimum_samples),
            "sample_ids": sample_ids,
            "inspection_context": _build_reference_metadata(config).get("inspection_context", {}),
        }
        _write_reference_metadata(metadata, artifact_paths["metadata"])
        result["rebuilt"] = True
        return result
    except Exception as exc:
        if artifact_paths["model"].exists():
            artifact_paths["model"].unlink()
        if artifact_paths["metadata"].exists():
            artifact_paths["metadata"].unlink()
        result["reason"] = f"Anomaly model training failed: {exc}"
        return result


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


def save_reference_metadata(
    config: dict,
    metadata_path: Optional[Path] = None,
    *,
    reference_role: str = "golden",
    extra_context: Optional[dict] = None,
) -> None:
    """Save reference creation settings as fingerprint for later validation."""
    try:
        meta_path = metadata_path
        if meta_path is None:
            active_paths = get_active_runtime_paths()
            meta_path = active_paths["reference_dir"] / "ref_meta.json"
        metadata = _build_reference_metadata(
            config,
            reference_role=reference_role,
            extra_context=extra_context or {"reference_id": "golden", "label": "Golden Reference", "state": "active"},
        )
        _write_reference_metadata(metadata, meta_path)
    except Exception as exc:
        # Don't crash on metadata save; just warn
        import sys
        print(f"Warning: Could not save reference metadata: {exc}", file=sys.stderr)


def load_reference_metadata(metadata_path: Optional[Path] = None) -> dict | None:
    """Load reference creation settings; returns None if metadata doesn't exist."""
    try:
        meta_path = metadata_path
        if meta_path is None:
            active_paths = get_active_runtime_paths()
            meta_path = active_paths["reference_dir"] / "ref_meta.json"
        return _load_reference_metadata_from_path(meta_path)
    except Exception:
        return None


def check_reference_settings_match(config: dict) -> tuple[bool, str | None]:
    """Compare current config against reference metadata. Returns (match, warning_msg)."""
    meta = load_reference_metadata()
    if meta is None:
        return True, None  # No metadata to compare
    
    inspection_cfg = config.get("inspection", {})
    alignment_cfg = config.get("alignment", {})
    mismatches: list[str] = []
    
    # Check ROI
    current_roi = _extract_roi_tuple(inspection_cfg)
    meta_roi = _extract_meta_roi_tuple(meta)
    if current_roi != meta_roi:
        mismatches.append(f"ROI mismatch: reference was {meta_roi}; current is {current_roi}")
    
    # Check threshold
    thresh_meta = meta.get("threshold", {})
    current_thresh_type = str(inspection_cfg.get("threshold_mode", "fixed")).lower()
    meta_thresh_type = str(thresh_meta.get("type", "fixed")).lower()
    if current_thresh_type != meta_thresh_type:
        mismatches.append(
            f"Threshold type mismatch: reference was {meta_thresh_type}; current is {current_thresh_type}"
        )
    
    current_thresh_val = float(inspection_cfg.get("threshold_value", 150.0))
    meta_thresh_val = float(thresh_meta.get("value", 150.0))
    if abs(current_thresh_val - meta_thresh_val) > 0.1:
        mismatches.append(
            f"Threshold value mismatch: reference was {meta_thresh_val}; current is {current_thresh_val}"
        )
    
    # Check morphology
    morph_meta = meta.get("morphology", {})
    current_ref_erode = int(inspection_cfg.get("reference_erode_iterations", 1))
    meta_ref_erode = int(morph_meta.get("reference_erode_iterations", 1))
    if current_ref_erode != meta_ref_erode:
        mismatches.append(f"Reference erode iterations mismatch: {meta_ref_erode} -> {current_ref_erode}")
    
    current_ref_dilate = int(inspection_cfg.get("reference_dilate_iterations", 1))
    meta_ref_dilate = int(morph_meta.get("reference_dilate_iterations", 1))
    if current_ref_dilate != meta_ref_dilate:
        mismatches.append(f"Reference dilate iterations mismatch: {meta_ref_dilate} -> {current_ref_dilate}")

    inspection_context = meta.get("inspection_context", {})
    inspection_checks = {
        "inspection_mode": str(inspection_cfg.get("inspection_mode", "mask_only")).lower(),
        "reference_strategy": str(inspection_cfg.get("reference_strategy", "golden_only")).lower(),
        "blend_mode": str(inspection_cfg.get("blend_mode", "hard_only")).lower(),
        "tolerance_mode": str(inspection_cfg.get("tolerance_mode", "balanced")).lower(),
    }
    for key, current_value in inspection_checks.items():
        meta_value = inspection_context.get(key)
        if meta_value is not None and str(meta_value).lower() != current_value:
            mismatches.append(f"{key.replace('_', ' ')} mismatch: reference was {meta_value}; current is {current_value}")

    alignment_meta = meta.get("alignment", {})
    current_profile = str(alignment_cfg.get("tolerance_profile", "balanced")).lower()
    meta_profile = alignment_meta.get("tolerance_profile")
    if meta_profile is not None and str(meta_profile).lower() != current_profile:
        mismatches.append(
            f"alignment tolerance profile mismatch: reference was {meta_profile}; current is {current_profile}"
        )

    if mismatches:
        return False, "; ".join(mismatches)
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

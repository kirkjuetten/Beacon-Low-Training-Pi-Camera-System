#!/usr/bin/env python3

import json
import time
from pathlib import Path
from typing import Callable, Optional

from inspection_system.app.camera_interface import get_active_runtime_paths
from inspection_system.app.registration_schema import build_alignment_metadata, get_registration_config
from inspection_system.app.reference_registration import build_registration_commissioning_summary
from inspection_system.app.segmentation import build_legacy_threshold_mode, resolve_segmentation_settings


def _load_reference_metadata_from_path(metadata_path: Path) -> dict | None:
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _write_reference_metadata(metadata: dict, metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


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


def _build_reference_metadata(
    config: dict,
    *,
    reference_role: str = "golden",
    extra_context: Optional[dict] = None,
) -> dict:
    inspection_cfg = config.get("inspection", {})
    segmentation_settings = resolve_segmentation_settings(inspection_cfg)
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
            "type": build_legacy_threshold_mode(segmentation_settings),
            "strategy": segmentation_settings["strategy_name"],
            "method": segmentation_settings["threshold_method"],
            "value": float(segmentation_settings["threshold_value"]),
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
        "alignment": build_alignment_metadata(config),
        "registration_baseline": build_registration_commissioning_summary(config),
        "reference_asset": {
            "role": str(reference_role).lower(),
        },
    }
    if extra_context:
        metadata["reference_asset"].update(extra_context)
    return metadata


def save_reference_metadata(
    config: dict,
    metadata_path: Optional[Path] = None,
    *,
    reference_role: str = "golden",
    extra_context: Optional[dict] = None,
) -> None:
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
        import sys
        print(f"Warning: Could not save reference metadata: {exc}", file=sys.stderr)


def load_reference_metadata(metadata_path: Optional[Path] = None) -> dict | None:
    try:
        meta_path = metadata_path
        if meta_path is None:
            active_paths = get_active_runtime_paths()
            meta_path = active_paths["reference_dir"] / "ref_meta.json"
        return _load_reference_metadata_from_path(meta_path)
    except Exception:
        return None


def check_reference_settings_match_impl(
    config: dict,
    *,
    metadata_loader: Callable[[], dict | None],
) -> tuple[bool, str | None]:
    meta = metadata_loader()
    if meta is None:
        return True, None

    inspection_cfg = config.get("inspection", {})
    alignment_cfg = config.get("alignment", {})
    mismatches: list[str] = []

    current_roi = _extract_roi_tuple(inspection_cfg)
    meta_roi = _extract_meta_roi_tuple(meta)
    if current_roi != meta_roi:
        mismatches.append(f"ROI mismatch: reference was {meta_roi}; current is {current_roi}")

    thresh_meta = meta.get("threshold", {})
    segmentation_settings = resolve_segmentation_settings(inspection_cfg)
    current_thresh_type = build_legacy_threshold_mode(segmentation_settings)
    meta_thresh_type = str(thresh_meta.get("type", "fixed")).lower()
    if current_thresh_type != meta_thresh_type:
        mismatches.append(
            f"Threshold type mismatch: reference was {meta_thresh_type}; current is {current_thresh_type}"
        )

    meta_strategy = thresh_meta.get("strategy")
    if meta_strategy is not None and str(meta_strategy).lower() != segmentation_settings["strategy_name"]:
        mismatches.append(
            "Segmentation strategy mismatch: "
            f"reference was {meta_strategy}; current is {segmentation_settings['strategy_name']}"
        )

    meta_method = thresh_meta.get("method")
    if meta_method is not None and str(meta_method).lower() != segmentation_settings["threshold_method"]:
        mismatches.append(
            "Threshold method mismatch: "
            f"reference was {meta_method}; current is {segmentation_settings['threshold_method']}"
        )

    current_thresh_val = float(segmentation_settings["threshold_value"])
    meta_thresh_val = float(thresh_meta.get("value", 150.0))
    if abs(current_thresh_val - meta_thresh_val) > 0.1:
        mismatches.append(
            f"Threshold value mismatch: reference was {meta_thresh_val}; current is {current_thresh_val}"
        )

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
    current_alignment_meta = build_alignment_metadata(config)
    current_registration = get_registration_config(config)
    current_profile = str(alignment_cfg.get("tolerance_profile", "balanced")).lower()
    meta_profile = alignment_meta.get("tolerance_profile")
    if meta_profile is not None and str(meta_profile).lower() != current_profile:
        mismatches.append(
            f"alignment tolerance profile mismatch: reference was {meta_profile}; current is {current_profile}"
        )

    meta_enabled = alignment_meta.get("enabled")
    if meta_enabled is not None and bool(meta_enabled) != current_alignment_meta["enabled"]:
        mismatches.append(
            f"alignment enabled mismatch: reference was {meta_enabled}; current is {current_alignment_meta['enabled']}"
        )

    meta_mode = alignment_meta.get("mode")
    if meta_mode is not None and str(meta_mode).lower() != current_alignment_meta["mode"]:
        mismatches.append(
            f"alignment mode mismatch: reference was {meta_mode}; current is {current_alignment_meta['mode']}"
        )

    meta_registration = alignment_meta.get("registration")
    if isinstance(meta_registration, dict):
        meta_registration_cfg = get_registration_config(
            {
                "alignment": {
                    "mode": alignment_meta.get("mode", current_alignment_meta["mode"]),
                    "registration": meta_registration,
                }
            }
        )
        registration_checks = {
            "registration strategy": (meta_registration_cfg["strategy"], current_registration["strategy"]),
            "registration transform model": (
                meta_registration_cfg["transform_model"],
                current_registration["transform_model"],
            ),
            "registration anchor mode": (meta_registration_cfg["anchor_mode"], current_registration["anchor_mode"]),
            "registration subpixel refinement": (
                meta_registration_cfg["subpixel_refinement"],
                current_registration["subpixel_refinement"],
            ),
            "registration search margin": (
                meta_registration_cfg["search_margin_px"],
                current_registration["search_margin_px"],
            ),
        }
        for label, (meta_value, current_value) in registration_checks.items():
            if meta_value != current_value:
                mismatches.append(f"{label} mismatch: reference was {meta_value}; current is {current_value}")

        if meta_registration_cfg["quality_gates"] != current_registration["quality_gates"]:
            mismatches.append(
                "registration quality gates mismatch: "
                f"reference was {meta_registration_cfg['quality_gates']}; current is {current_registration['quality_gates']}"
            )

        if meta_registration_cfg["datum_frame"] != current_registration["datum_frame"]:
            mismatches.append(
                "registration datum frame mismatch: "
                f"reference was {meta_registration_cfg['datum_frame']}; current is {current_registration['datum_frame']}"
            )

        if meta_registration_cfg["anchors"] != current_registration["anchors"]:
            mismatches.append(
                "registration anchors mismatch: "
                f"reference has {len(meta_registration_cfg['anchors'])}; current has {len(current_registration['anchors'])}"
            )

    if mismatches:
        return False, "; ".join(mismatches)
    return True, None
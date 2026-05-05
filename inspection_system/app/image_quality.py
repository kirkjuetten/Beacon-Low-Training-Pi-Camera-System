from __future__ import annotations

from pathlib import Path

from inspection_system.app.camera_interface import REFERENCE_MASK, get_active_runtime_paths
from inspection_system.app.reference_service import list_runtime_reference_candidates


def classify_invalid_capture(
    config: dict,
    image_path: Path,
    *,
    active_paths: dict | None = None,
    reference_candidates: list[dict] | None = None,
) -> str | None:
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
    width = int(roi_cfg.get("width", 0))
    height = int(roi_cfg.get("height", 0))

    if width > 0 and height > 0:
        if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
            return "Configured ROI is outside image bounds."

    runtime_paths = active_paths or get_active_runtime_paths()
    resolved_reference_candidates = (
        reference_candidates if reference_candidates is not None else list_runtime_reference_candidates(config, runtime_paths)
    )
    if not resolved_reference_candidates:
        reference_mask = Path(runtime_paths.get("reference_mask", REFERENCE_MASK))
        return f"Reference mask is missing: {reference_mask}"

    return None
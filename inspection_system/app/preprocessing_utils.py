from __future__ import annotations

from pathlib import Path

from inspection_system.app.segmentation import apply_segmentation_strategy


def _safe_int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_roi(image, roi_cfg: dict):
    x = _safe_int(roi_cfg.get("x", 0), 0)
    y = _safe_int(roi_cfg.get("y", 0), 0)
    w = _safe_int(roi_cfg.get("width", 0), 0)
    h = _safe_int(roi_cfg.get("height", 0), 0)

    if w <= 0 or h <= 0:
        return image, (0, 0, image.shape[1], image.shape[0])

    x2 = min(x + w, image.shape[1])
    y2 = min(y + h, image.shape[0])
    x = max(0, x)
    y = max(0, y)

    if x >= x2 or y >= y2:
        raise ValueError("Configured ROI is outside the image bounds.")

    return image[y:y2, x:x2], (x, y, x2 - x, y2 - y)


def _to_grayscale_float(image_or_gray, np_module):
    if image_or_gray is None:
        return None

    array = np_module.asarray(image_or_gray)
    if array.ndim == 2:
        return array.astype(np_module.float32, copy=False)

    if array.ndim == 3 and array.shape[2] >= 3:
        blue = array[:, :, 0].astype(np_module.float32, copy=False)
        green = array[:, :, 1].astype(np_module.float32, copy=False)
        red = array[:, :, 2].astype(np_module.float32, copy=False)
        return blue * 0.114 + green * 0.587 + red * 0.299

    return array.astype(np_module.float32, copy=False)


def _normalize_to_unit_interval(image, np_module):
    if image is None:
        return None

    image = image.astype(np_module.float32, copy=False)
    if image.size == 0:
        return image

    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value <= 1e-6:
        if max_value <= 1e-6:
            return np_module.zeros_like(image, dtype=np_module.float32)
        return np_module.clip(image / max_value, 0.0, 1.0).astype(np_module.float32, copy=False)

    return ((image - min_value) / (max_value - min_value)).astype(np_module.float32, copy=False)


def build_registration_image(image_or_gray, mask, np_module):
    gray = _to_grayscale_float(image_or_gray, np_module)
    if gray is None:
        raise ValueError("Registration image source is required.")

    gray_norm = _normalize_to_unit_interval(gray, np_module)
    gradient_x = np_module.abs(np_module.diff(gray_norm, axis=1, append=gray_norm[:, -1:]))
    gradient_y = np_module.abs(np_module.diff(gray_norm, axis=0, append=gray_norm[-1:, :]))
    gradient_norm = _normalize_to_unit_interval(gradient_x + gradient_y, np_module)

    if mask is None:
        mask_float = np_module.zeros_like(gray_norm, dtype=np_module.float32)
    else:
        mask_float = (np_module.asarray(mask) > 0).astype(np_module.float32, copy=False)
        if mask_float.shape != gray_norm.shape:
            mask_float = np_module.zeros_like(gray_norm, dtype=np_module.float32)

    masked_gray = gray_norm * (0.35 + (0.65 * mask_float))
    combined = (0.55 * masked_gray) + (0.30 * gradient_norm) + (0.15 * mask_float)
    registration_image = _normalize_to_unit_interval(combined, np_module)
    return np_module.rint(registration_image * 255.0).astype(np_module.uint8)


def make_binary_mask(image_path: Path, inspection_cfg: dict, import_cv2_and_numpy=None):
    # Backward-compatible fallback for older call sites that passed only (image_path, inspection_cfg).
    if import_cv2_and_numpy is None:
        from inspection_system.app.camera_interface import import_cv2_and_numpy as _import_cv2_and_numpy
        import_cv2_and_numpy = _import_cv2_and_numpy

    cv2, np = import_cv2_and_numpy()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    roi_image, roi = get_roi(image, inspection_cfg.get("roi", {}))
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    blur_kernel = int(inspection_cfg.get("blur_kernel", 3))
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    mask, _segmentation_settings = apply_segmentation_strategy(gray, inspection_cfg, cv2)

    return roi_image, gray, mask, roi, cv2, np
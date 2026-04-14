from __future__ import annotations

from pathlib import Path


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


def make_binary_mask(image_path: Path, inspection_cfg: dict, import_cv2_and_numpy):
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

    threshold_mode = str(inspection_cfg.get("threshold_mode", "fixed")).lower()
    threshold_value = int(inspection_cfg.get("threshold_value", 180))

    if threshold_mode == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_mode == "otsu_inv":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif threshold_mode == "fixed_inv":
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    else:
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    return roi_image, gray, mask, roi, cv2, np
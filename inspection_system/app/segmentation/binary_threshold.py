from __future__ import annotations


def apply_binary_threshold(gray, *, threshold_method: str, threshold_value: int, cv2):
    if threshold_method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return mask
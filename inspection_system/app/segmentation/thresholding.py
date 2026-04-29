"""Consolidated thresholding strategies.

Phase 5 of the refactor merges ``binary_threshold.py`` and
``binary_threshold_inverted.py`` into a single module so both flavors of
the same OpenCV threshold call live next to each other and a future
"adaptive" mode has an obvious place to land. The original modules are
preserved as thin shims that re-export from here, which keeps the eight
``apply_binary_threshold*`` import sites working.

Both functions accept ``cv2`` as a keyword argument so callers can inject
either the real binding or a mock during tests; this matches the existing
shape of :data:`SEGMENTATION_STRATEGIES`.
"""
from __future__ import annotations


def apply_binary_threshold(gray, *, threshold_method: str, threshold_value: int, cv2):
    """Standard binary threshold: bright pixels become foreground."""
    if threshold_method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return mask


def apply_binary_threshold_inverted(gray, *, threshold_method: str, threshold_value: int, cv2):
    """Inverted binary threshold: dark pixels become foreground."""
    if threshold_method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask

    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return mask

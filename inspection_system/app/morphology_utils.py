from __future__ import annotations


def erode_mask(mask, iterations: int, cv2, np):
    if iterations <= 0:
        return mask.copy()
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(mask, kernel, iterations=iterations)


def dilate_mask(mask, iterations: int, cv2, np):
    if iterations <= 0:
        return mask.copy()
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)
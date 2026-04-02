from __future__ import annotations


def get_mask_centroid_and_angle(mask, cv2, np):
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 5:
        return None, None

    moments = cv2.moments(mask, binaryImage=True)
    if abs(moments["m00"]) < 1e-6:
        return None, None

    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]

    coords = points.reshape(-1, 2).astype(np.float32)
    coords[:, 0] -= centroid_x
    coords[:, 1] -= centroid_y
    covariance = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    angle_deg = float(np.degrees(np.arctan2(principal_axis[1], principal_axis[0])))

    if angle_deg > 90.0:
        angle_deg -= 180.0
    elif angle_deg < -90.0:
        angle_deg += 180.0

    return (centroid_x, centroid_y), angle_deg


def rotate_mask(mask, angle_deg: float, cv2):
    height, width = mask.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def shift_mask(mask, shift_x: int, shift_y: int, cv2, np):
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    height, width = mask.shape[:2]
    return cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def align_sample_mask(sample_mask, reference_mask, alignment_cfg: dict, cv2, np):
    enabled = bool(alignment_cfg.get("enabled", True))
    if not enabled:
        return sample_mask, 0.0, 0, 0

    mode = str(alignment_cfg.get("mode", "moments")).lower()
    if mode != "moments":
        return sample_mask, 0.0, 0, 0

    ref_centroid, ref_angle = get_mask_centroid_and_angle(reference_mask, cv2, np)
    sample_centroid, sample_angle = get_mask_centroid_and_angle(sample_mask, cv2, np)
    if ref_centroid is None or ref_angle is None or sample_centroid is None or sample_angle is None:
        return sample_mask, 0.0, 0, 0

    max_angle_deg = float(alignment_cfg.get("max_angle_deg", 1.0))
    angle_delta = max(-max_angle_deg, min(max_angle_deg, ref_angle - sample_angle))

    rotated = rotate_mask(sample_mask, angle_delta, cv2) if abs(angle_delta) > 1e-6 else sample_mask
    rotated_centroid, _ = get_mask_centroid_and_angle(rotated, cv2, np)
    if rotated_centroid is None:
        return rotated, angle_delta, 0, 0

    shift_x = int(round(ref_centroid[0] - rotated_centroid[0]))
    shift_y = int(round(ref_centroid[1] - rotated_centroid[1]))
    max_shift_x = int(alignment_cfg.get("max_shift_x", 4))
    max_shift_y = int(alignment_cfg.get("max_shift_y", 3))
    shift_x = max(-max_shift_x, min(max_shift_x, shift_x))
    shift_y = max(-max_shift_y, min(max_shift_y, shift_y))

    aligned = shift_mask(rotated, shift_x, shift_y, cv2, np) if (shift_x != 0 or shift_y != 0) else rotated
    return aligned, float(angle_delta), int(shift_x), int(shift_y)
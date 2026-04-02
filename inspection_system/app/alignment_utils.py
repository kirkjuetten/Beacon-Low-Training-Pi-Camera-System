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
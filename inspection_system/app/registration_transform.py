from __future__ import annotations

import math


def get_transform_center(image_shape: tuple[int, int]) -> tuple[float, float]:
    height, width = image_shape[:2]
    return width / 2.0, height / 2.0


def apply_transform_to_point(
    point: tuple[float, float],
    image_shape: tuple[int, int],
    angle_deg: float,
    shift_x: int,
    shift_y: int,
) -> tuple[float, float]:
    center_x, center_y = get_transform_center(image_shape)
    angle_rad = math.radians(angle_deg)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    translated_x = point[0] - center_x
    translated_y = point[1] - center_y
    rotated_x = translated_x * cos_theta - translated_y * sin_theta
    rotated_y = translated_x * sin_theta + translated_y * cos_theta
    return rotated_x + center_x + shift_x, rotated_y + center_y + shift_y


def apply_inverse_transform_to_point(
    point: tuple[float, float],
    image_shape: tuple[int, int],
    angle_deg: float,
    shift_x: int,
    shift_y: int,
) -> tuple[float, float]:
    center_x, center_y = get_transform_center(image_shape)
    angle_rad = math.radians(-angle_deg)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    translated_x = point[0] - center_x - shift_x
    translated_y = point[1] - center_y - shift_y
    rotated_x = translated_x * cos_theta - translated_y * sin_theta
    rotated_y = translated_x * sin_theta + translated_y * cos_theta
    return rotated_x + center_x, rotated_y + center_y


def apply_transform_to_mask(mask, transform_summary: dict, cv2_module, np_module):
    height, width = mask.shape[:2]
    center = get_transform_center(mask.shape[:2])
    matrix = cv2_module.getRotationMatrix2D(center, float(transform_summary.get("angle_deg", 0.0)), 1.0)
    matrix[0, 2] += int(transform_summary.get("shift_x", 0))
    matrix[1, 2] += int(transform_summary.get("shift_y", 0))
    return cv2_module.warpAffine(
        mask,
        np_module.float32(matrix),
        (width, height),
        flags=cv2_module.INTER_NEAREST,
        borderMode=cv2_module.BORDER_CONSTANT,
        borderValue=0,
    )


def build_transform_summary(image_shape: tuple[int, int], angle_deg: float, shift_x: int, shift_y: int) -> dict:
    center_x, center_y = get_transform_center(image_shape)
    return {
        "angle_deg": float(angle_deg),
        "shift_x": int(shift_x),
        "shift_y": int(shift_y),
        "center": {
            "x": float(center_x),
            "y": float(center_y),
        },
    }
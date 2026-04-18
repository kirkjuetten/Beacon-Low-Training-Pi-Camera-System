from __future__ import annotations

from inspection_system.app.registration_transform import apply_transform_to_point, apply_inverse_transform_to_point


def _section_bounds(section_mask, np_module):
    points = np_module.argwhere(section_mask > 0)
    if points.size == 0:
        return None
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0) + 1
    center_x = float(points[:, 1].mean())
    center_y = float(points[:, 0].mean())
    return int(x0), int(y0), int(x1), int(y1), center_x, center_y


def compute_datum_section_measurements(sample_mask, section_masks, transform_summary: dict, np_module) -> dict:
    section_width_ratios = []
    section_center_offsets_px = []
    section_measurements = []
    sample_white = sample_mask > 0

    for section_mask in section_masks:
        bounds = _section_bounds(section_mask, np_module)
        if bounds is None:
            continue

        x0, y0, x1, y1, reference_center_x, reference_center_y = bounds
        reference_width = max(1, x1 - x0)
        reference_height = max(1, y1 - y0)
        reference_white = section_mask > 0
        reference_pixel_count = int(reference_white.sum())
        if reference_pixel_count == 0:
            continue
        reference_center = (reference_center_x, reference_center_y)
        reference_corners = [
            (float(x0), float(y0)),
            (float(x1), float(y0)),
            (float(x0), float(y1)),
            (float(x1), float(y1)),
        ]
        expected_sample_corners = [
            apply_inverse_transform_to_point(
                corner,
                sample_mask.shape[:2],
                float(transform_summary.get("angle_deg", 0.0)),
                int(transform_summary.get("shift_x", 0)),
                int(transform_summary.get("shift_y", 0)),
            )
            for corner in reference_corners
        ]
        sample_xs = [point[0] for point in expected_sample_corners]
        sample_ys = [point[1] for point in expected_sample_corners]
        padding_x = max(1, int(np_module.ceil(reference_width / 2.0)))
        padding_y = max(1, int(np_module.ceil(reference_height / 2.0)))
        window_x0 = max(0, int(np_module.floor(min(sample_xs))) - padding_x)
        window_y0 = max(0, int(np_module.floor(min(sample_ys))) - padding_y)
        window_x1 = min(sample_mask.shape[1], int(np_module.ceil(max(sample_xs))) + padding_x)
        window_y1 = min(sample_mask.shape[0], int(np_module.ceil(max(sample_ys))) + padding_y)

        if window_x1 <= window_x0 or window_y1 <= window_y0:
            continue

        window = sample_white[window_y0:window_y1, window_x0:window_x1]
        sample_points = np_module.argwhere(window)
        if sample_points.size == 0:
            section_width_ratios.append(0.0)
            section_center_offsets_px.append(float("inf"))
            section_measurements.append(
                {
                    "reference_center": {"x": float(reference_center[0]), "y": float(reference_center[1])},
                    "expected_sample_window": {
                        "x": int(window_x0),
                        "y": int(window_y0),
                        "width": int(window_x1 - window_x0),
                        "height": int(window_y1 - window_y0),
                    },
                    "sample_detected": False,
                }
            )
            continue

        sample_points[:, 0] += window_y0
        sample_points[:, 1] += window_x0
        transformed_points = np_module.array(
            [
                apply_transform_to_point(
                    (float(point[1]), float(point[0])),
                    sample_mask.shape[:2],
                    float(transform_summary.get("angle_deg", 0.0)),
                    int(transform_summary.get("shift_x", 0)),
                    int(transform_summary.get("shift_y", 0)),
                )
                for point in sample_points
            ],
            dtype=np_module.float32,
        )
        transformed_x = np_module.rint(transformed_points[:, 0]).astype(np_module.int32)
        transformed_y = np_module.rint(transformed_points[:, 1]).astype(np_module.int32)
        valid_points = (
            (transformed_x >= 0)
            & (transformed_x < sample_mask.shape[1])
            & (transformed_y >= 0)
            & (transformed_y < sample_mask.shape[0])
        )
        if not valid_points.any():
            section_width_ratios.append(0.0)
            section_center_offsets_px.append(float("inf"))
            section_measurements.append(
                {
                    "reference_center": {"x": float(reference_center[0]), "y": float(reference_center[1])},
                    "expected_sample_window": {
                        "x": int(window_x0),
                        "y": int(window_y0),
                        "width": int(window_x1 - window_x0),
                        "height": int(window_y1 - window_y0),
                    },
                    "sample_detected": False,
                }
            )
            continue

        transformed_points = transformed_points[valid_points]
        transformed_x = transformed_x[valid_points]
        transformed_y = transformed_y[valid_points]
        in_section = (
            (transformed_x >= x0)
            & (transformed_x < x1)
            & (transformed_y >= y0)
            & (transformed_y < y1)
        )
        section_points_reference = transformed_points[in_section] if in_section.any() else transformed_points
        observed_center_reference = (
            float(section_points_reference[:, 0].mean()),
            float(section_points_reference[:, 1].mean()),
        )
        observed_width = float(section_points_reference[:, 0].max() - section_points_reference[:, 0].min() + 1.0)
        observed_pixel_count = int(section_points_reference.shape[0])
        width_ratio = observed_pixel_count / float(reference_pixel_count)
        center_offset = abs(observed_center_reference[0] - reference_center[0])

        section_width_ratios.append(width_ratio)
        section_center_offsets_px.append(center_offset)
        section_measurements.append(
            {
                "reference_center": {"x": float(reference_center[0]), "y": float(reference_center[1])},
                "expected_sample_window": {
                    "x": int(window_x0),
                    "y": int(window_y0),
                    "width": int(window_x1 - window_x0),
                    "height": int(window_y1 - window_y0),
                },
                "sample_detected": True,
                "observed_center_reference": {
                    "x": float(observed_center_reference[0]),
                    "y": float(observed_center_reference[1]),
                },
                "observed_width_px": observed_width,
                "observed_pixel_count": observed_pixel_count,
                "reference_pixel_count": reference_pixel_count,
                "reference_width_px": float(reference_width),
                "width_ratio": float(width_ratio),
                "center_offset_px": float(center_offset),
            }
        )

    return {
        "section_width_ratios": section_width_ratios,
        "section_center_offsets_px": section_center_offsets_px,
        "section_measurements": section_measurements,
        "frame": "datum",
    }
from __future__ import annotations

import numpy as np

from capture_test import get_roi, score_sample


def test_get_roi_returns_configured_crop_and_bounds() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    roi_image, roi = get_roi(image, {"x": 10, "y": 20, "width": 50, "height": 30})

    assert roi == (10, 20, 50, 30)
    assert roi_image.shape == (30, 50, 3)


def test_get_roi_rejects_out_of_bounds_region() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    try:
        get_roi(image, {"x": 250, "y": 20, "width": 50, "height": 30})
    except ValueError as exc:
        assert "outside the image bounds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for invalid ROI")


def test_score_sample_computes_expected_metrics() -> None:
    reference_allowed = np.array(
        [
            [255, 255, 255],
            [255, 255, 255],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    reference_required = np.array(
        [
            [255, 255, 0],
            [255, 255, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    sample_mask = np.array(
        [
            [255, 255, 255],
            [255, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    section_masks = [
        np.array(
            [
                [255, 0, 0],
                [255, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [0, 255, 0],
                [0, 255, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    ]

    metrics = score_sample(reference_allowed, reference_required, sample_mask, section_masks)

    assert metrics["sample_white_pixels"] == 4
    assert metrics["required_coverage"] == 0.75
    assert metrics["outside_allowed_ratio"] == 0.0
    assert metrics["min_section_coverage"] == 0.5
    assert metrics["section_coverages"] == [1.0, 0.5]

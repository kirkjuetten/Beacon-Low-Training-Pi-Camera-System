from pathlib import Path

import numpy as np

from preprocessing_utils import build_registration_image, get_roi, make_binary_mask
from inspection_system.app.segmentation import resolve_segmentation_settings


class FakeCv2:
    IMREAD_COLOR = 1
    COLOR_BGR2GRAY = 2
    THRESH_BINARY = 4
    THRESH_BINARY_INV = 16
    THRESH_OTSU = 8

    def __init__(self, image):
        self.image = image
        self.calls = []

    def imread(self, path, mode):
        self.calls.append(("imread", path, mode))
        return self.image

    def cvtColor(self, roi_image, color_code):
        self.calls.append(("cvtColor", roi_image.shape, color_code))
        return roi_image[:, :, 0]

    def GaussianBlur(self, gray, kernel, sigma):
        self.calls.append(("GaussianBlur", gray.shape, kernel, sigma))
        return gray

    def threshold(self, gray, threshold_value, max_value, threshold_mode):
        self.calls.append(("threshold", gray.shape, threshold_value, max_value, threshold_mode))
        return 0, np.full_like(gray, 255)


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
    else:
        raise AssertionError("Expected ValueError for invalid ROI")


def test_make_binary_mask_uses_roi_blur_and_threshold() -> None:
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    fake_cv2 = FakeCv2(image)

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    roi_image, gray, mask, roi, _, _ = make_binary_mask(
        Path("sample.jpg"),
        {
            "roi": {"x": 1, "y": 1, "width": 2, "height": 2},
            "blur_kernel": 3,
            "threshold_mode": "fixed",
            "threshold_value": 180,
        },
        fake_import_cv2_and_numpy,
    )

    assert roi == (1, 1, 2, 2)
    assert roi_image.shape == (2, 2, 3)
    assert gray.shape == (2, 2)
    assert mask.shape == (2, 2)

    assert fake_cv2.calls[0] == ("imread", "sample.jpg", fake_cv2.IMREAD_COLOR)
    assert fake_cv2.calls[1] == ("cvtColor", (2, 2, 3), fake_cv2.COLOR_BGR2GRAY)
    assert fake_cv2.calls[2] == ("GaussianBlur", (2, 2), (3, 3), 0)
    assert fake_cv2.calls[3] == ("threshold", (2, 2), 180, 255, fake_cv2.THRESH_BINARY)


def test_make_binary_mask_supports_otsu_inv() -> None:
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    fake_cv2 = FakeCv2(image)

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    make_binary_mask(
        Path("sample.jpg"),
        {
            "roi": {"x": 0, "y": 0, "width": 2, "height": 2},
            "blur_kernel": 1,
            "threshold_mode": "otsu_inv",
        },
        fake_import_cv2_and_numpy,
    )

    assert fake_cv2.calls[2] == (
        "threshold",
        (2, 2),
        0,
        255,
        fake_cv2.THRESH_BINARY_INV + fake_cv2.THRESH_OTSU,
    )


def test_make_binary_mask_supports_fixed_inv() -> None:
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    fake_cv2 = FakeCv2(image)

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    make_binary_mask(
        Path("sample.jpg"),
        {
            "roi": {"x": 0, "y": 0, "width": 2, "height": 2},
            "blur_kernel": 1,
            "threshold_mode": "fixed_inv",
            "threshold_value": 123,
        },
        fake_import_cv2_and_numpy,
    )

    assert fake_cv2.calls[2] == (
        "threshold",
        (2, 2),
        123,
        255,
        fake_cv2.THRESH_BINARY_INV,
    )


def test_make_binary_mask_supports_explicit_segmentation_strategy() -> None:
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    fake_cv2 = FakeCv2(image)

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    make_binary_mask(
        Path("sample.jpg"),
        {
            "roi": {"x": 0, "y": 0, "width": 2, "height": 2},
            "blur_kernel": 1,
            "segmentation_strategy": "binary_threshold_inverted",
            "threshold_method": "otsu",
        },
        fake_import_cv2_and_numpy,
    )

    assert fake_cv2.calls[2] == (
        "threshold",
        (2, 2),
        0,
        255,
        fake_cv2.THRESH_BINARY_INV + fake_cv2.THRESH_OTSU,
    )


def test_resolve_segmentation_settings_maps_legacy_threshold_modes() -> None:
    settings = resolve_segmentation_settings({"threshold_mode": "fixed_inv", "threshold_value": 123})

    assert settings == {
        "strategy_name": "binary_threshold_inverted",
        "threshold_method": "fixed",
        "threshold_value": 123,
    }


def test_build_registration_image_emphasizes_masked_feature_signal() -> None:
    gray = np.array(
        [
            [10, 10, 10, 10, 10],
            [10, 25, 90, 25, 10],
            [10, 35, 140, 35, 10],
            [10, 25, 90, 25, 10],
            [10, 10, 10, 10, 10],
        ],
        dtype=np.uint8,
    )
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255

    registration_image = build_registration_image(gray, mask, np)

    assert registration_image.shape == gray.shape
    assert registration_image.dtype == np.uint8
    assert int(registration_image[2, 2]) > int(registration_image[0, 0])
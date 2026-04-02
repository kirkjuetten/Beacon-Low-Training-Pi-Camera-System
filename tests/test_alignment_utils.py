import numpy as np

from alignment_utils import get_mask_centroid_and_angle, rotate_mask, shift_mask


class FakeCv2Rotate:
    INTER_NEAREST = "INTER_NEAREST"
    BORDER_CONSTANT = "BORDER_CONSTANT"

    def __init__(self):
        self.calls = []

    def getRotationMatrix2D(self, center, angle_deg, scale):
        self.calls.append(("getRotationMatrix2D", center, angle_deg, scale))
        return "rotation-matrix"

    def warpAffine(self, mask, matrix, size, flags=None, borderMode=None, borderValue=None):
        self.calls.append(("warpAffine", matrix, size, flags, borderMode, borderValue))
        return "rotated-mask"


class FakeCv2Shift:
    INTER_NEAREST = "INTER_NEAREST"
    BORDER_CONSTANT = "BORDER_CONSTANT"

    def __init__(self):
        self.calls = []

    def warpAffine(self, mask, matrix, size, flags=None, borderMode=None, borderValue=None):
        self.calls.append(("warpAffine", matrix, size, flags, borderMode, borderValue))
        return "shifted-mask"


class FakeCv2Points:
    def __init__(self, points):
        self._points = points

    def findNonZero(self, _mask):
        return self._points


def test_get_mask_centroid_and_angle_returns_none_for_insufficient_points() -> None:
    fake_cv2 = FakeCv2Points(
        np.array([[[0, 0]], [[1, 1]], [[2, 2]], [[3, 3]]], dtype=np.int32)
    )
    mask = np.zeros((5, 5), dtype=np.uint8)

    centroid, angle = get_mask_centroid_and_angle(mask, fake_cv2, np)

    assert centroid is None
    assert angle is None


def test_rotate_mask_calls_cv2_with_expected_geometry() -> None:
    fake_cv2 = FakeCv2Rotate()
    mask = np.zeros((4, 6), dtype=np.uint8)

    result = rotate_mask(mask, 12.5, fake_cv2)

    assert result == "rotated-mask"
    assert fake_cv2.calls[0] == ("getRotationMatrix2D", (3.0, 2.0), 12.5, 1.0)
    assert fake_cv2.calls[1] == (
        "warpAffine",
        "rotation-matrix",
        (6, 4),
        "INTER_NEAREST",
        "BORDER_CONSTANT",
        0,
    )


def test_shift_mask_calls_cv2_with_expected_translation() -> None:
    fake_cv2 = FakeCv2Shift()
    mask = np.zeros((3, 5), dtype=np.uint8)

    result = shift_mask(mask, 2, -1, fake_cv2, np)

    assert result == "shifted-mask"
    call = fake_cv2.calls[0]
    assert call[0] == "warpAffine"
    assert call[2] == (5, 3)
    assert call[3] == "INTER_NEAREST"
    assert call[4] == "BORDER_CONSTANT"
    assert call[5] == 0

    matrix = call[1]
    assert matrix.shape == (2, 3)
    assert matrix[0, 2] == 2
    assert matrix[1, 2] == -1
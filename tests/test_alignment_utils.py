import numpy as np
import alignment_utils
from alignment_utils import align_sample_mask, get_mask_centroid_and_angle, rotate_mask, shift_mask


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


def test_align_sample_mask_returns_original_when_disabled() -> None:
    sample_mask = "sample-mask"
    reference_mask = "reference-mask"

    result = align_sample_mask(sample_mask, reference_mask, {"enabled": False}, object(), np)

    assert result == ("sample-mask", 0.0, 0, 0)


def test_align_sample_mask_returns_original_when_mode_is_not_moments() -> None:
    sample_mask = "sample-mask"
    reference_mask = "reference-mask"

    result = align_sample_mask(
        sample_mask,
        reference_mask,
        {"enabled": True, "mode": "off"},
        object(),
        np,
    )

    assert result == ("sample-mask", 0.0, 0, 0)


def test_align_sample_mask_applies_clamped_rotation_and_shift(monkeypatch) -> None:
    calls = []

    def fake_get_mask_centroid_and_angle(mask, cv2, np_module):
        if mask == "reference-mask":
            return (10.0, 10.0), 5.0
        if mask == "sample-mask":
            return (2.0, 3.0), 1.0
        if mask == "rotated-mask":
            return (0.0, 0.0), 0.0
        raise AssertionError(f"Unexpected mask: {mask}")

    def fake_rotate_mask(mask, angle_deg, cv2):
        calls.append(("rotate", mask, angle_deg))
        return "rotated-mask"

    def fake_shift_mask(mask, shift_x, shift_y, cv2, np_module):
        calls.append(("shift", mask, shift_x, shift_y))
        return "aligned-mask"

    monkeypatch.setattr(alignment_utils, "get_mask_centroid_and_angle", fake_get_mask_centroid_and_angle)
    monkeypatch.setattr(alignment_utils, "rotate_mask", fake_rotate_mask)
    monkeypatch.setattr(alignment_utils, "shift_mask", fake_shift_mask)

    result = align_sample_mask(
        "sample-mask",
        "reference-mask",
        {
            "enabled": True,
            "mode": "moments",
            "max_angle_deg": 1.5,
            "max_shift_x": 4,
            "max_shift_y": 3,
        },
        object(),
        np,
    )

    assert result == ("aligned-mask", 1.5, 4, 3)
    assert calls == [
        ("rotate", "sample-mask", 1.5),
        ("shift", "rotated-mask", 4, 3),
    ]
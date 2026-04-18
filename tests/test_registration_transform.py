import numpy as np

from inspection_system.app.registration_transform import (
    apply_inverse_transform_to_point,
    apply_transform_to_mask,
    apply_transform_to_point,
    build_transform_summary,
)


class FakeCv2:
    INTER_NEAREST = "INTER_NEAREST"
    BORDER_CONSTANT = "BORDER_CONSTANT"

    def __init__(self):
        self.calls = []

    def getRotationMatrix2D(self, center, angle_deg, scale):
        self.calls.append(("getRotationMatrix2D", center, angle_deg, scale))
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def warpAffine(self, mask, matrix, size, flags=None, borderMode=None, borderValue=None):
        self.calls.append(("warpAffine", matrix.copy(), size, flags, borderMode, borderValue))
        return mask


def test_apply_transform_to_point_rotates_around_image_center_and_shifts() -> None:
    transformed = apply_transform_to_point((10.0, 20.0), (30, 30), 90.0, 2, -1)

    assert transformed == (12.0, 9.0)


def test_build_transform_summary_returns_serializable_transform_fields() -> None:
    summary = build_transform_summary((20, 40), 12.5, 3, -2)

    assert summary == {
        "angle_deg": 12.5,
        "shift_x": 3,
        "shift_y": -2,
        "center": {"x": 20.0, "y": 10.0},
    }


def test_apply_inverse_transform_to_point_reverses_forward_transform() -> None:
    point = (10.0, 20.0)
    transformed = apply_transform_to_point(point, (30, 30), 90.0, 2, -1)
    restored = apply_inverse_transform_to_point(transformed, (30, 30), 90.0, 2, -1)

    assert restored == point


def test_apply_transform_to_mask_applies_rotation_and_translation() -> None:
    mask = np.zeros((10, 12), dtype=np.uint8)
    fake_cv2 = FakeCv2()

    transformed = apply_transform_to_mask(
        mask,
        {"angle_deg": 7.5, "shift_x": 3, "shift_y": -2},
        fake_cv2,
        np,
    )

    assert transformed is mask
    assert fake_cv2.calls[0] == ("getRotationMatrix2D", (6.0, 5.0), 7.5, 1.0)
    assert fake_cv2.calls[1][0] == "warpAffine"
    np.testing.assert_array_equal(fake_cv2.calls[1][1], np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], dtype=np.float32))
    assert fake_cv2.calls[1][2] == (12, 10)
    assert fake_cv2.calls[1][3] == "INTER_NEAREST"
    assert fake_cv2.calls[1][4] == "BORDER_CONSTANT"
    assert fake_cv2.calls[1][5] == 0
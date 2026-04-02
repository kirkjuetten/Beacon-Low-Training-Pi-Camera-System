import numpy as np

from morphology_utils import dilate_mask, erode_mask


class FakeCv2:
    def __init__(self):
        self.calls = []

    def erode(self, mask, kernel, iterations=1):
        self.calls.append(("erode", mask, kernel, iterations))
        return "eroded-mask"

    def dilate(self, mask, kernel, iterations=1):
        self.calls.append(("dilate", mask, kernel, iterations))
        return "dilated-mask"


def test_erode_mask_returns_copy_when_iterations_non_positive() -> None:
    fake_cv2 = FakeCv2()
    mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    result = erode_mask(mask, 0, fake_cv2, np)

    assert np.array_equal(result, mask)
    assert result is not mask
    assert fake_cv2.calls == []


def test_dilate_mask_returns_copy_when_iterations_non_positive() -> None:
    fake_cv2 = FakeCv2()
    mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    result = dilate_mask(mask, -1, fake_cv2, np)

    assert np.array_equal(result, mask)
    assert result is not mask
    assert fake_cv2.calls == []


def test_erode_mask_calls_cv2_with_expected_kernel() -> None:
    fake_cv2 = FakeCv2()
    mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    result = erode_mask(mask, 2, fake_cv2, np)

    assert result == "eroded-mask"
    call = fake_cv2.calls[0]
    assert call[0] == "erode"
    assert call[3] == 2
    kernel = call[2]
    assert kernel.shape == (3, 3)
    assert np.all(kernel == 1)


def test_dilate_mask_calls_cv2_with_expected_kernel() -> None:
    fake_cv2 = FakeCv2()
    mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    result = dilate_mask(mask, 3, fake_cv2, np)

    assert result == "dilated-mask"
    call = fake_cv2.calls[0]
    assert call[0] == "dilate"
    assert call[3] == 3
    kernel = call[2]
    assert kernel.shape == (3, 3)
    assert np.all(kernel == 1)
import numpy as np

from section_mask_utils import compute_section_masks


class FakeCv2:
    CC_STAT_LEFT = 0
    CC_STAT_TOP = 1
    CC_STAT_WIDTH = 2
    CC_STAT_HEIGHT = 3

    def __init__(self, labels, stats):
        self._labels = labels
        self._stats = stats

    def connectedComponentsWithStats(self, _white, connectivity=8):
        return 2, self._labels, self._stats, None


def test_compute_section_masks_splits_wide_component() -> None:
    required_mask = np.full((2, 24), 255, dtype=np.uint8)
    labels = np.ones((2, 24), dtype=np.int32)
    stats = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 24, 2, 48],
        ],
        dtype=np.int32,
    )
    fake_cv2 = FakeCv2(labels, stats)

    sections = compute_section_masks(required_mask, 3, fake_cv2, np)

    assert len(sections) == 3
    assert [int((section > 0).sum()) for section in sections] == [16, 16, 16]


def test_compute_section_masks_keeps_small_component_as_one_section() -> None:
    required_mask = np.full((2, 6), 255, dtype=np.uint8)
    labels = np.ones((2, 6), dtype=np.int32)
    stats = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 6, 2, 12],
        ],
        dtype=np.int32,
    )
    fake_cv2 = FakeCv2(labels, stats)

    sections = compute_section_masks(required_mask, 12, fake_cv2, np)

    assert len(sections) == 1
    assert int((sections[0] > 0).sum()) == 12
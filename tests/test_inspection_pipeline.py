from pathlib import Path

import numpy as np

from inspection_pipeline import inspect_against_reference


class FakeCv2:
    IMREAD_GRAYSCALE = 1

    def __init__(self, reference_mask):
        self.reference_mask = reference_mask
        self.calls = []

    def imread(self, path, mode):
        self.calls.append(("imread", path, mode))
        return self.reference_mask


def test_inspect_against_reference_returns_expected_details() -> None:
    sample_mask = np.array([[255, 255], [255, 255]], dtype=np.uint8)
    reference_mask = np.array([[255, 255], [255, 255]], dtype=np.uint8)
    fake_cv2 = FakeCv2(reference_mask)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return None, None, sample_mask, (1, 2, 3, 4), fake_cv2, np

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        return sample_mask_arg, 1.25, 2, -1

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        allowed = np.array([[255, 255], [255, 255]], dtype=np.uint8)
        required = np.array([[255, 255], [255, 255]], dtype=np.uint8)
        return allowed, required

    def fake_compute_section_masks(required_mask_arg, section_columns, cv2, np_module):
        return [required_mask_arg]

    def fake_score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks):
        return {
            "required_coverage": 0.95,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.90,
            "section_coverages": [0.90],
            "sample_white_pixels": 4,
            "missing_required_mask": np.zeros((2, 2), dtype=bool),
            "outside_allowed_mask": np.zeros((2, 2), dtype=bool),
        }

    def fake_evaluate_metrics(metrics, inspection_cfg):
        return True, {
            "required_coverage": 0.95,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.90,
            "min_required_coverage": 0.92,
            "max_outside_allowed_ratio": 0.02,
            "min_section_coverage_limit": 0.85,
        }

    def fake_save_debug_outputs(stem, aligned_sample_mask, diff):
        return {"mask": "mask-path", "diff": "diff-path"}

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    def fake_dilate_mask(mask, iterations, cv2, np_module):
        return mask

    def fake_erode_mask(mask, iterations, cv2, np_module):
        return mask

    passed, details = inspect_against_reference(
        {"inspection": {"save_debug_images": True}, "alignment": {}},
        Path("sample.jpg"),
        fake_make_binary_mask,
        Path("reference.png"),
        fake_align_sample_mask,
        fake_build_reference_regions,
        fake_compute_section_masks,
        fake_score_sample,
        fake_evaluate_metrics,
        fake_save_debug_outputs,
        fake_import_cv2_and_numpy,
        fake_dilate_mask,
        fake_erode_mask,
    )

    assert passed is True
    assert details["roi"] == {"x": 1, "y": 2, "width": 3, "height": 4}
    assert details["best_angle_deg"] == 1.25
    assert details["best_shift_x"] == 2
    assert details["best_shift_y"] == -1
    assert details["required_coverage"] == 0.95
    assert details["outside_allowed_ratio"] == 0.01
    assert details["min_section_coverage"] == 0.90
    assert details["section_coverages"] == [0.90]
    assert details["sample_white_pixels"] == 4
    assert details["min_required_coverage"] == 0.92
    assert details["max_outside_allowed_ratio"] == 0.02
    assert details["min_section_coverage_limit"] == 0.85
    assert details["debug_paths"] == {"mask": "mask-path", "diff": "diff-path"}


def test_inspect_against_reference_raises_for_shape_mismatch() -> None:
    sample_mask = np.zeros((2, 2), dtype=np.uint8)
    reference_mask = np.zeros((3, 3), dtype=np.uint8)
    fake_cv2 = FakeCv2(reference_mask)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return None, None, sample_mask, (0, 0, 2, 2), fake_cv2, np

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    try:
        inspect_against_reference(
            {"inspection": {}, "alignment": {}},
            Path("sample.jpg"),
            fake_make_binary_mask,
            Path("reference.png"),
            lambda *args: None,
            lambda *args: None,
            lambda *args: None,
            lambda *args: None,
            lambda *args: None,
            lambda *args: None,
            fake_import_cv2_and_numpy,
            lambda mask, iterations, cv2, np_module: mask,
            lambda mask, iterations, cv2, np_module: mask,
        )
    except ValueError as exc:
        assert "does not match sample mask shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched reference and sample mask shapes")
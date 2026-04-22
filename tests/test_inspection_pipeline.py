from pathlib import Path
from unittest import mock

import numpy as np

import inspection_pipeline
from inspection_pipeline import (
    compute_mean_edge_distance_px,
    compute_section_center_offsets_px,
    compute_section_edge_distances_px,
    compute_section_width_ratios,
    inspect_against_reference,
    inspect_against_references,
)


class FakeCv2:
    IMREAD_GRAYSCALE = 0  
    IMREAD_COLOR = 1
    INTER_NEAREST = "INTER_NEAREST"
    BORDER_CONSTANT = "BORDER_CONSTANT"

    def __init__(self, grayscale_image, color_image=None):
        self.grayscale_image = grayscale_image
        self.color_image = color_image if color_image is not None else grayscale_image
        self.calls = []

    def imread(self, path, mode):
        self.calls.append(("imread", path, mode))
        if mode == self.IMREAD_GRAYSCALE:
            return self.grayscale_image
        else:
            return self.color_image

    def getRotationMatrix2D(self, center, angle_deg, scale):
        self.calls.append(("getRotationMatrix2D", center, angle_deg, scale))
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def warpAffine(self, mask, matrix, size, flags=None, borderMode=None, borderValue=None):
        self.calls.append(("warpAffine", matrix, size, flags, borderMode, borderValue))
        return mask


def test_inspect_against_reference_returns_expected_details() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[5:15, 5:15] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[5:15, 5:15] = 255
    roi_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    fake_cv2 = FakeCv2(reference_mask, roi_image)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return roi_image, None, sample_mask, (1, 2, 3, 4), fake_cv2, np

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        return sample_mask_arg, 1.25, 2, -1

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        allowed = np.zeros((20, 20), dtype=np.uint8)
        allowed[5:15, 5:15] = 255
        required = np.zeros((20, 20), dtype=np.uint8)
        required[5:15, 5:15] = 255
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
            "missing_required_mask": np.zeros((20, 20), dtype=bool),
            "outside_allowed_mask": np.zeros((20, 20), dtype=bool),
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

    with mock.patch('inspection_system.app.anomaly_detection_utils.compute_ssim', return_value=0.95):
        with mock.patch('inspection_system.app.anomaly_detection_utils.compute_histogram_similarity', return_value=0.92):
            passed, details = inspect_against_reference(
                {"inspection": {"save_debug_images": True}, "alignment": {}},
                Path("sample.jpg"),
                fake_make_binary_mask,
                Path("reference.png"),
                Path("reference_image.png"),
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
    assert details["registration"]["status"] == "aligned"
    assert details["registration"]["runtime_mode"] == "moments"
    assert details["registration"]["requested_strategy"] == "moments"
    assert details["registration"]["applied_strategy"] == "moments"
    assert details["registration"]["transform"]["angle_deg"] == 1.25
    assert details["registration"]["observed_anchors"] == []
    assert details["edge_measurement_frame"] == "datum"
    assert details["required_coverage"] == 0.95
    assert details["outside_allowed_ratio"] == 0.01
    assert details["min_section_coverage"] == 0.90
    assert details["mean_edge_distance_px"] == 0.0
    assert details["worst_section_edge_distance_px"] == 0.0
    assert details["section_coverages"] == [0.90]
    assert details["sample_white_pixels"] == 4
    assert details["min_required_coverage"] == 0.92
    assert details["max_outside_allowed_ratio"] == 0.02
    assert details["min_section_coverage_limit"] == 0.85
    assert details["debug_paths"] == {"mask": "mask-path", "diff": "diff-path"}


def test_inspect_against_reference_skips_debug_diff_when_debug_images_disabled() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[5:15, 5:15] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[5:15, 5:15] = 255
    roi_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    fake_cv2 = FakeCv2(reference_mask, roi_image)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return roi_image, None, sample_mask, (1, 2, 3, 4), fake_cv2, np

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        return sample_mask_arg, 1.25, 2, -1

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        return reference_mask_arg, reference_mask_arg

    def fake_compute_section_masks(required_mask_arg, section_columns, cv2, np_module):
        return [required_mask_arg]

    def fake_score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks):
        return {
            "required_coverage": 0.95,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.90,
            "section_coverages": [0.90],
            "sample_white_pixels": 4,
            "missing_required_mask": np.zeros((20, 20), dtype=bool),
            "outside_allowed_mask": np.zeros((20, 20), dtype=bool),
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

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    with mock.patch.object(
        inspection_pipeline,
        "_build_debug_diff",
        side_effect=AssertionError("debug diff should not be built when debug output is disabled"),
    ):
        passed, details = inspect_against_reference(
            {"inspection": {"save_debug_images": False}, "alignment": {}},
            Path("sample.jpg"),
            fake_make_binary_mask,
            Path("reference.png"),
            Path("reference_image.png"),
            fake_align_sample_mask,
            fake_build_reference_regions,
            fake_compute_section_masks,
            fake_score_sample,
            fake_evaluate_metrics,
            lambda stem, aligned_sample_mask, diff: {"mask": "mask-path", "diff": "diff-path"},
            fake_import_cv2_and_numpy,
            lambda mask, iterations, cv2, np_module: mask,
            lambda mask, iterations, cv2, np_module: mask,
        )

    assert passed is True
    assert details["debug_paths"] == {}


def test_inspect_against_reference_applies_authoritative_feature_gates() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[5:15, 5:15] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[5:15, 5:15] = 255
    roi_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    fake_cv2 = FakeCv2(reference_mask, roi_image)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return roi_image, None, sample_mask, (0, 0, 20, 20), fake_cv2, np

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        return sample_mask_arg, 0.0, 0, 0

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        return reference_mask_arg, reference_mask_arg

    def fake_compute_section_masks(required_mask_arg, section_columns, cv2, np_module):
        return [required_mask_arg]

    def fake_score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks):
        return {
            "required_coverage": 0.95,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.90,
            "section_coverages": [0.90],
            "sample_white_pixels": 4,
            "missing_required_mask": np.zeros((20, 20), dtype=bool),
            "outside_allowed_mask": np.zeros((20, 20), dtype=bool),
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

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    feature_measurements = [
        {
            "feature_key": "isolated_feature_1",
            "feature_label": "Isolated Feature 1",
            "feature_family": "isolated_centroid",
            "feature_type": "isolated_centroid_position",
            "measurement_frame": "datum",
            "sample_detected": True,
            "failure_cause": "feature_position",
            "dx_px": 1.6,
            "dy_px": 0.1,
            "radial_offset_px": 1.603121954,
            "center_offset_px": 1.603121954,
        }
    ]
    feature_summary = {
        "feature_key": "isolated_feature_1",
        "feature_label": "Isolated Feature 1",
        "feature_family": "isolated_centroid",
        "feature_type": "isolated_centroid_position",
        "measurement_frame": "datum",
        "feature_count": 1,
        "sample_detected": True,
        "failure_cause": "feature_position",
        "dx_px": 1.6,
        "dy_px": 0.1,
        "radial_offset_px": 1.603121954,
        "center_offset_px": 1.603121954,
    }

    with mock.patch.object(
        inspection_pipeline,
        "extract_molded_part_feature_measurements",
        return_value=(feature_measurements, feature_summary),
    ):
        with mock.patch('inspection_system.app.anomaly_detection_utils.compute_ssim', return_value=0.95):
            with mock.patch('inspection_system.app.anomaly_detection_utils.compute_histogram_similarity', return_value=0.92):
                passed, details = inspect_against_reference(
                    {
                        "inspection": {
                            "save_debug_images": False,
                            "feature_position_families": ["isolated_centroid"],
                            "feature_gate_thresholds": {"max_dx_px": 1.0, "max_radial_offset_px": 2.0},
                        },
                        "alignment": {},
                    },
                    Path("sample.jpg"),
                    fake_make_binary_mask,
                    Path("reference.png"),
                    Path("reference_image.png"),
                    fake_align_sample_mask,
                    fake_build_reference_regions,
                    fake_compute_section_masks,
                    fake_score_sample,
                    fake_evaluate_metrics,
                    lambda stem, aligned_sample_mask, diff: {},
                    fake_import_cv2_and_numpy,
                    lambda mask, iterations, cv2, np_module: mask,
                    lambda mask, iterations, cv2, np_module: mask,
                )

    assert passed is False
    assert details["inspection_failure_cause"] == "feature_position"
    assert details["feature_gate_active"] is True
    assert details["feature_gate_passed"] is False
    assert details["feature_gate_metric"] == "dx_px"
    assert details["max_feature_dx_px"] == 1.0
    assert details["feature_position_summary"]["feature_key"] == "isolated_feature_1"


def test_inspect_against_reference_aggregates_multiple_authoritative_lanes() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[5:15, 5:15] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[5:15, 5:15] = 255
    roi_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    fake_cv2 = FakeCv2(reference_mask, roi_image)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return roi_image, None, sample_mask, (0, 0, 20, 20), fake_cv2, np

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        return sample_mask_arg, 0.0, 0, 0

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        return reference_mask_arg, reference_mask_arg

    def fake_compute_section_masks(required_mask_arg, section_columns, cv2, np_module):
        return [required_mask_arg]

    def fake_score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks):
        lane_id = None
        return {
            "required_coverage": 0.98,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.95,
            "section_coverages": [0.95],
            "sample_white_pixels": 25,
            "missing_required_mask": np.zeros((20, 20), dtype=bool),
            "outside_allowed_mask": np.zeros((20, 20), dtype=bool),
        }

    def fake_evaluate_metrics(metrics, inspection_cfg):
        lane_id = inspection_cfg.get("lane_id")
        passed = lane_id != "print"
        return passed, {
            "required_coverage": 0.98 if lane_id != "print" else 0.88,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.95,
            "min_required_coverage": 0.92 if lane_id != "print" else 0.9,
            "max_outside_allowed_ratio": 0.02,
            "min_section_coverage_limit": 0.85,
            "inspection_mode": "mask_only",
        }

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    with mock.patch('inspection_system.app.anomaly_detection_utils.compute_ssim', return_value=0.95):
        with mock.patch('inspection_system.app.anomaly_detection_utils.compute_histogram_similarity', return_value=0.92):
            passed, details = inspect_against_reference(
                {
                    "inspection_program": {
                        "program_id": "multi_lane_program",
                        "lanes": [
                            {"lane_id": "geometry", "lane_type": "measurement", "authoritative": True},
                            {
                                "lane_id": "print",
                                "lane_type": "measurement",
                                "authoritative": True,
                                "inspection": {"lane_tag": "print"},
                            },
                        ],
                    },
                    "inspection": {"save_debug_images": False},
                    "alignment": {},
                },
                Path("sample.jpg"),
                fake_make_binary_mask,
                Path("reference.png"),
                Path("reference_image.png"),
                fake_align_sample_mask,
                fake_build_reference_regions,
                fake_compute_section_masks,
                fake_score_sample,
                fake_evaluate_metrics,
                lambda stem, aligned_sample_mask, diff: {},
                fake_import_cv2_and_numpy,
                lambda mask, iterations, cv2, np_module: mask,
                lambda mask, iterations, cv2, np_module: mask,
            )

    assert passed is False
    assert details["inspection_program"]["program_id"] == "multi_lane_program"
    assert details["inspection_program"]["primary_lane_id"] == "geometry"
    assert details["inspection_program"]["active_lane_id"] == "print"
    assert details["failed_lane_ids"] == ["print"]
    assert details["failed_authoritative_lane_ids"] == ["print"]
    assert len(details["lane_results"]) == 2
    assert details["lane_results"][0]["lane_id"] == "geometry"
    assert details["lane_results"][1]["lane_id"] == "print"
    assert details["required_coverage"] == 0.88


def test_compute_mean_edge_distance_px_detects_shifted_edges() -> None:
    reference_mask = np.zeros((12, 12), dtype=np.uint8)
    sample_mask = np.zeros((12, 12), dtype=np.uint8)
    reference_mask[3:9, 3:9] = 255
    sample_mask[3:9, 4:10] = 255

    mean_edge_distance_px = compute_mean_edge_distance_px(reference_mask, sample_mask, np)

    assert mean_edge_distance_px > 0.0


def test_compute_section_edge_distances_px_reports_per_section_drift() -> None:
    reference_mask = np.zeros((12, 12), dtype=np.uint8)
    sample_mask = np.zeros((12, 12), dtype=np.uint8)
    reference_mask[3:9, 3:9] = 255
    sample_mask[3:9, 3:9] = 255
    sample_mask[3:9, 6:10] = 255
    left_section = np.zeros((12, 12), dtype=np.uint8)
    right_section = np.zeros((12, 12), dtype=np.uint8)
    left_section[:, :6] = 255
    right_section[:, 6:] = 255

    distances = compute_section_edge_distances_px(reference_mask, sample_mask, [left_section, right_section], np)

    assert len(distances) == 2
    assert distances[0] < distances[1]


def test_compute_section_width_ratios_detects_narrower_section() -> None:
    reference_mask = np.zeros((12, 12), dtype=np.uint8)
    sample_mask = np.zeros((12, 12), dtype=np.uint8)
    reference_mask[3:9, 3:9] = 255
    sample_mask[3:9, 3:8] = 255
    left_section = np.zeros((12, 12), dtype=np.uint8)
    right_section = np.zeros((12, 12), dtype=np.uint8)
    left_section[3:9, 3:6] = 255
    right_section[3:9, 6:9] = 255

    ratios = compute_section_width_ratios(reference_mask, sample_mask, [left_section, right_section], np)

    assert len(ratios) == 2
    assert ratios[0] == 1.0
    assert ratios[1] < 1.0


def test_compute_section_center_offsets_px_detects_shifted_section_center() -> None:
    reference_mask = np.zeros((12, 12), dtype=np.uint8)
    sample_mask = np.zeros((12, 12), dtype=np.uint8)
    reference_mask[3:9, 3:9] = 255
    sample_mask[3:9, 4:10] = 255
    left_section = np.zeros((12, 12), dtype=np.uint8)
    right_section = np.zeros((12, 12), dtype=np.uint8)
    left_section[3:9, 3:6] = 255
    right_section[3:9, 6:9] = 255

    offsets = compute_section_center_offsets_px(reference_mask, sample_mask, [left_section, right_section], np)

    assert len(offsets) == 2
    assert offsets[0] > 0.0
    assert offsets[1] >= 0.0
    assert max(offsets) > 0.0


def test_inspect_against_reference_raises_for_shape_mismatch() -> None:
    sample_mask = np.zeros((2, 2), dtype=np.uint8)
    reference_mask = np.zeros((3, 3), dtype=np.uint8)
    reference_image = np.zeros((3, 3, 3), dtype=np.uint8)
    fake_cv2 = FakeCv2(reference_mask, reference_image)

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
            Path("reference_image.png"),
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


def test_inspect_against_reference_applies_alignment_profile_defaults() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[5:15, 5:15] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[5:15, 5:15] = 255
    roi_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    fake_cv2 = FakeCv2(reference_mask, roi_image)
    seen_alignment_cfg = {}

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return roi_image, None, sample_mask, (0, 0, 20, 20), fake_cv2, np

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        seen_alignment_cfg.update(alignment_cfg)
        return sample_mask_arg, 0.0, 0, 0

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        return reference_mask_arg, reference_mask_arg

    def fake_compute_section_masks(required_mask_arg, section_columns, cv2, np_module):
        return [required_mask_arg]

    def fake_score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks):
        return {
            "required_coverage": 0.95,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.90,
            "section_coverages": [0.90],
            "sample_white_pixels": 4,
            "missing_required_mask": np.zeros((20, 20), dtype=bool),
            "outside_allowed_mask": np.zeros((20, 20), dtype=bool),
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

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    with mock.patch('inspection_system.app.anomaly_detection_utils.compute_ssim', return_value=0.95):
        with mock.patch('inspection_system.app.anomaly_detection_utils.compute_histogram_similarity', return_value=0.92):
            passed, details = inspect_against_reference(
                {"inspection": {"save_debug_images": False}, "alignment": {"tolerance_profile": "forgiving"}},
                Path("sample.jpg"),
                fake_make_binary_mask,
                Path("reference.png"),
                Path("reference_image.png"),
                fake_align_sample_mask,
                fake_build_reference_regions,
                fake_compute_section_masks,
                fake_score_sample,
                fake_evaluate_metrics,
                lambda stem, aligned_sample_mask, diff: {},
                fake_import_cv2_and_numpy,
                lambda mask, iterations, cv2, np_module: mask,
                lambda mask, iterations, cv2, np_module: mask,
            )

    assert passed is True
    assert details["alignment_profile"] == "forgiving"
    assert details["registration"]["runtime_mode"] == "moments"
    assert seen_alignment_cfg["max_angle_deg"] == 1.8
    assert seen_alignment_cfg["max_shift_x"] == 7
    assert seen_alignment_cfg["max_shift_y"] == 5


def test_inspect_against_reference_reports_staged_registration_strategy_when_runtime_remains_moments() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[5:15, 5:15] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[5:15, 5:15] = 255
    roi_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    fake_cv2 = FakeCv2(reference_mask, roi_image)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return roi_image, None, sample_mask, (0, 0, 20, 20), fake_cv2, np

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        return sample_mask_arg, 0.0, 0, 0

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        return reference_mask_arg, reference_mask_arg

    def fake_compute_section_masks(required_mask_arg, section_columns, cv2, np_module):
        return [required_mask_arg]

    def fake_score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks):
        return {
            "required_coverage": 0.95,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.90,
            "section_coverages": [0.90],
            "sample_white_pixels": 4,
            "missing_required_mask": np.zeros((20, 20), dtype=bool),
            "outside_allowed_mask": np.zeros((20, 20), dtype=bool),
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

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    with mock.patch('inspection_system.app.anomaly_detection_utils.compute_ssim', return_value=0.95):
        with mock.patch('inspection_system.app.anomaly_detection_utils.compute_histogram_similarity', return_value=0.92):
            passed, details = inspect_against_reference(
                {
                    "inspection": {"save_debug_images": False},
                    "alignment": {
                        "mode": "moments",
                        "registration": {
                            "strategy": "anchor_pair",
                            "transform_model": "similarity",
                            "anchor_mode": "pair",
                            "subpixel_refinement": "phase_correlation",
                        },
                    },
                },
                Path("sample.jpg"),
                fake_make_binary_mask,
                Path("reference.png"),
                Path("reference_image.png"),
                fake_align_sample_mask,
                fake_build_reference_regions,
                fake_compute_section_masks,
                fake_score_sample,
                fake_evaluate_metrics,
                lambda stem, aligned_sample_mask, diff: {},
                fake_import_cv2_and_numpy,
                lambda mask, iterations, cv2, np_module: mask,
                lambda mask, iterations, cv2, np_module: mask,
            )

    assert passed is True
    assert details["registration"]["requested_strategy"] == "anchor_pair"
    assert details["registration"]["applied_strategy"] == "moments"
    assert details["registration"]["fallback_reason"] == (
        "Requested registration strategy 'anchor_pair' is staged but runtime is using 'moments'."
    )


def test_inspect_against_reference_supports_anchor_translation_runtime_summary() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[8:10, 7:9] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[8:10, 9:11] = 255
    roi_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    fake_cv2 = FakeCv2(reference_mask, roi_image)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return roi_image, None, sample_mask, (0, 0, 20, 20), fake_cv2, np

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        return reference_mask_arg, reference_mask_arg

    def fake_compute_section_masks(required_mask_arg, section_columns, cv2, np_module):
        return [required_mask_arg]

    def fake_score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks):
        assert aligned_sample_mask.shape == sample_mask.shape
        return {
            "required_coverage": 0.95,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.90,
            "section_coverages": [0.90],
            "sample_white_pixels": 4,
            "missing_required_mask": np.zeros((20, 20), dtype=bool),
            "outside_allowed_mask": np.zeros((20, 20), dtype=bool),
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

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    with mock.patch('inspection_system.app.anomaly_detection_utils.compute_ssim', return_value=0.95):
        with mock.patch('inspection_system.app.anomaly_detection_utils.compute_histogram_similarity', return_value=0.92):
            passed, details = inspect_against_reference(
                {
                    "inspection": {"save_debug_images": False},
                    "alignment": {
                        "mode": "anchor_translation",
                        "max_shift_x": 10,
                        "max_shift_y": 10,
                        "registration": {
                            "strategy": "anchor_translation",
                            "anchor_mode": "single",
                            "anchors": [
                                {
                                    "anchor_id": "anchor_a",
                                    "reference_point": {"x": 10, "y": 10},
                                    "search_window": {"x": 6, "y": 7, "width": 6, "height": 6},
                                }
                            ],
                        },
                    },
                },
                Path("sample.jpg"),
                fake_make_binary_mask,
                Path("reference.png"),
                Path("reference_image.png"),
                lambda *args: (_ for _ in ()).throw(AssertionError("moments aligner should not be called")),
                fake_build_reference_regions,
                fake_compute_section_masks,
                fake_score_sample,
                fake_evaluate_metrics,
                lambda stem, aligned_sample_mask, diff: {},
                fake_import_cv2_and_numpy,
                lambda mask, iterations, cv2, np_module: mask,
                lambda mask, iterations, cv2, np_module: mask,
            )

    assert passed is True
    assert details["registration"]["runtime_mode"] == "anchor_translation"
    assert details["registration"]["applied_strategy"] == "anchor_translation"
    assert details["best_shift_x"] == 2
    assert details["best_shift_y"] == 2
    assert details["registration"]["quality"]["confidence"] > 0.0
    assert details["registration"]["transform"]["shift_x"] == 2
    assert details["registration"]["observed_anchors"][0]["anchor_id"] == "anchor_a"
    assert details["edge_measurement_frame"] == "datum"
    assert details["section_measurement_frame"] == "datum"
    assert details["feature_measurements"][0]["feature_type"] == "datum_section_position"
    assert details["feature_measurements"][0]["section_index"] == 0
    assert details["feature_position_summary"]["feature_key"] == "section_1"
    assert details["section_measurements"][0]["sample_detected"] is True


def test_inspect_against_references_selects_best_passing_reference() -> None:
    def fake_inspect_against_reference(
        config,
        image_path,
        make_binary_mask,
        reference_mask_path,
        reference_image_path,
        *args,
        anomaly_detector=None,
    ):
        ref_name = Path(reference_mask_path).stem
        if ref_name == 'candidate_mask':
            return True, {
                'required_coverage': 0.96,
                'outside_allowed_ratio': 0.01,
                'min_section_coverage': 0.93,
                'effective_min_required_coverage': 0.9,
                'effective_max_outside_allowed_ratio': 0.02,
                'effective_min_section_coverage': 0.85,
            }
        return False, {
            'required_coverage': 0.86,
            'outside_allowed_ratio': 0.03,
            'min_section_coverage': 0.81,
            'effective_min_required_coverage': 0.9,
            'effective_max_outside_allowed_ratio': 0.02,
            'effective_min_section_coverage': 0.85,
        }

    with mock.patch.object(inspection_pipeline, 'inspect_against_reference', side_effect=fake_inspect_against_reference):
        passed, details = inspect_against_references(
            {'inspection': {'reference_strategy': 'hybrid'}},
            Path('sample.jpg'),
            [
                {
                    'reference_id': 'golden',
                    'label': 'Golden Reference',
                    'role': 'golden',
                    'reference_mask_path': Path('golden_mask.png'),
                    'reference_image_path': Path('golden_image.png'),
                },
                {
                    'reference_id': 'candidate_1',
                    'label': 'Approved Good 1',
                    'role': 'candidate',
                    'reference_mask_path': Path('candidate_mask.png'),
                    'reference_image_path': Path('candidate_image.png'),
                },
            ],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    assert passed is True
    assert details['reference_id'] == 'candidate_1'
    assert details['reference_label'] == 'Approved Good 1'
    assert len(details['reference_candidate_summaries']) == 2
    assert details['reference_candidate_summaries'][0]['reference_id'] == 'golden'
    assert details['reference_candidate_summaries'][1]['reference_id'] == 'candidate_1'
    assert details['reference_candidate_summaries'][1]['passed'] is True


def test_reference_candidate_rank_ignores_nonfinite_optional_gate_values() -> None:
    rank = inspection_pipeline._reference_candidate_rank(
        False,
        {
            'required_coverage': 0.95,
            'effective_min_required_coverage': 0.92,
            'outside_allowed_ratio': 0.03,
            'effective_max_outside_allowed_ratio': 0.02,
            'min_section_coverage': 0.9,
            'effective_min_section_coverage': 0.85,
            'section_center_gate_active': False,
            'section_width_gate_active': False,
            'section_edge_gate_active': False,
            'edge_distance_gate_active': False,
            'worst_section_center_offset_px': float('inf'),
            'effective_max_section_center_offset_px': float('inf'),
        },
    )

    assert rank == (0, -1, 0.07)


def test_should_prefer_coarse_moments_measurement_for_width_drift_false_accept_pattern() -> None:
    refined_measurement = {
        'passed': True,
        'threshold_summary': {
            'required_coverage': 0.985027,
            'outside_allowed_ratio': 0.00635,
            'mean_edge_distance_px': 2.324482,
            'worst_section_width_delta_ratio': 1.0,
            'effective_max_section_width_delta_ratio': 1.0,
        },
    }
    coarse_measurement = {
        'passed': False,
        'threshold_summary': {
            'required_coverage': 0.977109,
            'outside_allowed_ratio': 0.009386,
            'mean_edge_distance_px': 2.93541,
            'worst_section_width_delta_ratio': 1.75,
            'effective_max_section_width_delta_ratio': 1.0,
        },
    }

    assert inspection_pipeline._should_prefer_coarse_moments_measurement(
        refined_measurement,
        coarse_measurement,
        {'angle_deg': 0.224514, 'shift_x': -4, 'shift_y': -2},
        {'angle_deg': 0.474514, 'shift_x': -4, 'shift_y': -3},
    ) is True


def test_should_keep_refined_moments_measurement_when_independent_metrics_improve_enough() -> None:
    refined_measurement = {
        'passed': True,
        'threshold_summary': {
            'required_coverage': 0.992199,
            'outside_allowed_ratio': 0.005243,
            'mean_edge_distance_px': 1.853271,
            'worst_section_width_delta_ratio': 1.0,
            'effective_max_section_width_delta_ratio': 1.0,
        },
    }
    coarse_measurement = {
        'passed': False,
        'threshold_summary': {
            'required_coverage': 0.983119,
            'outside_allowed_ratio': 0.01156,
            'mean_edge_distance_px': 2.667853,
            'worst_section_width_delta_ratio': 2.0,
            'effective_max_section_width_delta_ratio': 1.0,
        },
    }

    assert inspection_pipeline._should_prefer_coarse_moments_measurement(
        refined_measurement,
        coarse_measurement,
        {'angle_deg': 0.036225, 'shift_x': -4, 'shift_y': 0},
        {'angle_deg': 0.286225, 'shift_x': -4, 'shift_y': -1},
    ) is False


def test_inspect_against_reference_uses_anomaly_detector_metrics_in_runtime_decision() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[5:15, 5:15] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[5:15, 5:15] = 255
    roi_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    fake_cv2 = FakeCv2(reference_mask, roi_image)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return roi_image, None, sample_mask, (0, 0, 20, 20), fake_cv2, np

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        return sample_mask_arg, 0.0, 0, 0

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        return reference_mask_arg, reference_mask_arg

    def fake_compute_section_masks(required_mask_arg, section_columns, cv2, np_module):
        return [required_mask_arg]

    def fake_score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks):
        return {
            'required_coverage': 0.96,
            'outside_allowed_ratio': 0.01,
            'min_section_coverage': 0.92,
            'section_coverages': [0.92],
            'sample_white_pixels': 25,
            'missing_required_mask': np.zeros((20, 20), dtype=bool),
            'outside_allowed_mask': np.zeros((20, 20), dtype=bool),
        }

    def fake_evaluate_metrics(metrics, inspection_cfg):
        assert metrics['mean_edge_distance_px'] == 0.0
        assert metrics['worst_section_edge_distance_px'] == 0.0
        assert metrics['worst_section_width_delta_ratio'] == 0.0
        assert metrics['worst_section_center_offset_px'] == 0.0
        assert metrics['anomaly_score'] == 0.25
        return False, {
            'required_coverage': 0.96,
            'outside_allowed_ratio': 0.01,
            'min_section_coverage': 0.92,
            'min_required_coverage': 0.92,
            'max_outside_allowed_ratio': 0.02,
            'min_section_coverage_limit': 0.85,
            'mean_edge_distance_px': 0.0,
            'max_mean_edge_distance_px': 1.0,
            'effective_max_mean_edge_distance_px': 1.0,
            'worst_section_edge_distance_px': 0.0,
            'max_section_edge_distance_px': 0.8,
            'effective_max_section_edge_distance_px': 0.8,
            'worst_section_width_delta_ratio': 0.0,
            'max_section_width_delta_ratio': 0.15,
            'effective_max_section_width_delta_ratio': 0.15,
            'worst_section_center_offset_px': 0.0,
            'max_section_center_offset_px': 0.75,
            'effective_max_section_center_offset_px': 0.75,
            'min_anomaly_score': 0.5,
            'effective_min_anomaly_score': 0.5,
            'inspection_mode': 'mask_and_ml',
            'edge_distance_gate_active': True,
            'section_edge_gate_active': True,
            'section_width_gate_active': True,
            'section_center_gate_active': True,
            'anomaly_gate_active': True,
        }

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    with mock.patch.object(inspection_pipeline, 'detect_anomalies', return_value={'ssim': 0.95, 'histogram_similarity': 0.9, 'mse': 1.0, 'anomaly_score': 0.25}):
        passed, details = inspect_against_reference(
            {'inspection': {'inspection_mode': 'mask_and_ml', 'save_debug_images': False}, 'alignment': {}},
            Path('sample.jpg'),
            fake_make_binary_mask,
            Path('reference.png'),
            Path('reference_image.png'),
            fake_align_sample_mask,
            fake_build_reference_regions,
            fake_compute_section_masks,
            fake_score_sample,
            fake_evaluate_metrics,
            lambda stem, aligned_sample_mask, diff: {},
            fake_import_cv2_and_numpy,
            lambda mask, iterations, cv2, np_module: mask,
            lambda mask, iterations, cv2, np_module: mask,
            anomaly_detector=object(),
        )

    assert passed is False
    assert details['mean_edge_distance_px'] == 0.0
    assert details['edge_distance_gate_active'] is True
    assert details['worst_section_edge_distance_px'] == 0.0
    assert details['section_edge_gate_active'] is True
    assert details['worst_section_width_delta_ratio'] == 0.0
    assert details['section_width_gate_active'] is True
    assert details['worst_section_center_offset_px'] == 0.0
    assert details['section_center_gate_active'] is True
    assert details['anomaly_score'] == 0.25
    assert details['anomaly_gate_active'] is True


def test_inspect_against_reference_uses_aligned_mask_measurements_when_registration_is_not_aligned() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[5:15, 5:15] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[5:15, 5:15] = 255
    roi_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    fake_cv2 = FakeCv2(reference_mask, roi_image)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return roi_image, None, sample_mask, (0, 0, 20, 20), fake_cv2, np

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        return reference_mask_arg, reference_mask_arg

    def fake_compute_section_masks(required_mask_arg, section_columns, cv2, np_module):
        return [required_mask_arg]

    def fake_score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks):
        return {
            'required_coverage': 0.96,
            'outside_allowed_ratio': 0.01,
            'min_section_coverage': 0.92,
            'section_coverages': [0.92],
            'sample_white_pixels': 25,
            'missing_required_mask': np.zeros((20, 20), dtype=bool),
            'outside_allowed_mask': np.zeros((20, 20), dtype=bool),
        }

    def fake_evaluate_metrics(metrics, inspection_cfg):
        assert metrics['worst_section_width_delta_ratio'] == 0.0
        assert metrics['worst_section_center_offset_px'] == 0.0
        return True, {
            'required_coverage': 0.96,
            'outside_allowed_ratio': 0.01,
            'min_section_coverage': 0.92,
            'min_required_coverage': 0.92,
            'max_outside_allowed_ratio': 0.02,
            'min_section_coverage_limit': 0.85,
        }

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        assert alignment_cfg['mode'] == 'moments'
        return sample_mask_arg, 0.0, 0, 0

    passed, details = inspect_against_reference(
        {
            'inspection': {'save_debug_images': False},
            'alignment': {
                'mode': 'rigid_refined',
                'registration': {'strategy': 'rigid_refined', 'subpixel_refinement': 'template'},
            },
        },
        Path('sample.jpg'),
        fake_make_binary_mask,
        Path('reference.png'),
        Path('reference_image.png'),
        fake_align_sample_mask,
        fake_build_reference_regions,
        fake_compute_section_masks,
        fake_score_sample,
        fake_evaluate_metrics,
        lambda stem, aligned_sample_mask, diff: {},
        fake_import_cv2_and_numpy,
        lambda mask, iterations, cv2, np_module: mask,
        lambda mask, iterations, cv2, np_module: mask,
    )

    assert passed is True
    assert details['registration']['status'] == 'aligned'
    assert details['registration']['applied_strategy'] == 'rigid_refined'
    assert details['registration']['subpixel_refinement'] == 'template'
    assert details['edge_measurement_frame'] == 'datum'
    assert details['section_measurement_frame'] == 'datum'


def test_inspect_against_reference_rejects_when_registration_quality_gate_fails() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[8:10, 7:9] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[8:10, 9:11] = 255
    roi_image = np.ones((20, 20, 3), dtype=np.uint8) * 128
    fake_cv2 = FakeCv2(reference_mask, roi_image)

    def fake_make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy):
        return roi_image, None, sample_mask, (0, 0, 20, 20), fake_cv2, np

    def fake_build_reference_regions(reference_mask_arg, inspection_cfg, dilate_fn, erode_fn):
        return reference_mask_arg, reference_mask_arg

    def fake_compute_section_masks(required_mask_arg, section_columns, cv2, np_module):
        return [required_mask_arg]

    def fake_score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks):
        return {
            'required_coverage': 0.95,
            'outside_allowed_ratio': 0.01,
            'min_section_coverage': 0.90,
            'section_coverages': [0.90],
            'sample_white_pixels': 4,
            'missing_required_mask': np.zeros((20, 20), dtype=bool),
            'outside_allowed_mask': np.zeros((20, 20), dtype=bool),
        }

    def fake_evaluate_metrics(metrics, inspection_cfg):
        return True, {
            'required_coverage': 0.95,
            'outside_allowed_ratio': 0.01,
            'min_section_coverage': 0.90,
            'min_required_coverage': 0.92,
            'max_outside_allowed_ratio': 0.02,
            'min_section_coverage_limit': 0.85,
        }

    def fake_import_cv2_and_numpy():
        return fake_cv2, np

    with mock.patch.object(
        inspection_pipeline,
        'detect_anomalies',
        side_effect=AssertionError('detect_anomalies should not run when registration rejects'),
    ):
        passed, details = inspect_against_reference(
            {
                'inspection': {
                    'inspection_mode': 'mask_and_ml',
                    'min_anomaly_score': 0.5,
                    'save_debug_images': False,
                },
                'alignment': {
                    'mode': 'anchor_translation',
                    'max_shift_x': 10,
                    'max_shift_y': 10,
                    'registration': {
                        'strategy': 'anchor_translation',
                        'anchor_mode': 'single',
                        'quality_gates': {'min_confidence': 1.1},
                        'anchors': [
                            {
                                'anchor_id': 'anchor_a',
                                'reference_point': {'x': 10, 'y': 10},
                                'search_window': {'x': 6, 'y': 7, 'width': 6, 'height': 6},
                            }
                        ],
                    },
                },
            },
            Path('sample.jpg'),
            fake_make_binary_mask,
            Path('reference.png'),
            Path('reference_image.png'),
            lambda *args: (_ for _ in ()).throw(AssertionError('moments aligner should not be called')),
            fake_build_reference_regions,
            fake_compute_section_masks,
            fake_score_sample,
            fake_evaluate_metrics,
            lambda stem, aligned_sample_mask, diff: {},
            fake_import_cv2_and_numpy,
            lambda mask, iterations, cv2, np_module: mask,
            lambda mask, iterations, cv2, np_module: mask,
            anomaly_detector=object(),
        )

    assert passed is False
    assert details['registration']['status'] == 'quality_gate_failed'
    assert details['registration']['rejection_reason'] is not None
    assert details['registration']['quality_gate_failures'][0]['gate_key'] == 'min_confidence'
    assert details['failure_stage'] == 'registration'
    assert details['inspection_failure_cause'] == 'registration_failure'
    assert details['edge_measurement_frame'] == 'aligned_mask'
    assert details['section_measurement_frame'] == 'aligned_mask'
    assert details['feature_measurements'] == []
    assert details['feature_position_summary'] is None
    assert 'anomaly_score' not in details
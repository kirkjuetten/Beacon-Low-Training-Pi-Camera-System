from pathlib import Path
from unittest import mock

import numpy as np

import inspection_pipeline
from inspection_pipeline import (
    compute_mean_edge_distance_px,
    compute_section_edge_distances_px,
    compute_section_width_ratios,
    inspect_against_reference,
    inspect_against_references,
)


class FakeCv2:
    IMREAD_GRAYSCALE = 0  
    IMREAD_COLOR = 1

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
    assert seen_alignment_cfg["max_angle_deg"] == 1.8
    assert seen_alignment_cfg["max_shift_x"] == 7
    assert seen_alignment_cfg["max_shift_y"] == 5


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
            'min_anomaly_score': 0.5,
            'effective_min_anomaly_score': 0.5,
            'inspection_mode': 'mask_and_ml',
            'edge_distance_gate_active': True,
            'section_edge_gate_active': True,
            'section_width_gate_active': True,
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
    assert details['anomaly_score'] == 0.25
    assert details['anomaly_gate_active'] is True
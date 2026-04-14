from __future__ import annotations

import numpy as np

from scoring_utils import evaluate_metrics, score_sample


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


def test_evaluate_metrics_applies_thresholds() -> None:
    metrics = {
        "required_coverage": 0.95,
        "outside_allowed_ratio": 0.01,
        "min_section_coverage": 0.9,
    }
    inspection_cfg = {
        "min_required_coverage": 0.92,
        "max_outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.85,
    }

    passed, summary = evaluate_metrics(metrics, inspection_cfg)

    assert passed is True
    assert summary["required_coverage"] == 0.95
    assert summary["outside_allowed_ratio"] == 0.01
    assert summary["min_section_coverage"] == 0.9
    assert summary["min_required_coverage"] == 0.92
    assert summary["max_outside_allowed_ratio"] == 0.02
    assert summary["min_section_coverage_limit"] == 0.85


def test_evaluate_metrics_applies_optional_ssim_and_mse_gates() -> None:
    metrics = {
        "required_coverage": 0.95,
        "outside_allowed_ratio": 0.01,
        "min_section_coverage": 0.9,
        "ssim": 0.91,
        "mse": 4.0,
    }
    inspection_cfg = {
        "min_required_coverage": 0.92,
        "max_outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.85,
        "min_ssim": 0.9,
        "max_mse": 5.0,
    }

    passed, summary = evaluate_metrics(metrics, inspection_cfg)

    assert passed is True
    assert summary["ssim"] == 0.91
    assert summary["mse"] == 4.0
    assert summary["min_ssim"] == 0.9
    assert summary["max_mse"] == 5.0


def test_evaluate_metrics_fails_when_optional_gate_is_not_met() -> None:
    metrics = {
        "required_coverage": 0.95,
        "outside_allowed_ratio": 0.01,
        "min_section_coverage": 0.9,
        "ssim": 0.75,
        "mse": 9.0,
        "anomaly_score": -0.2,
    }
    inspection_cfg = {
        "min_required_coverage": 0.92,
        "max_outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.85,
        "min_ssim": 0.9,
        "max_mse": 5.0,
        "min_anomaly_score": 0.0,
    }

    passed, summary = evaluate_metrics(metrics, inspection_cfg)

    assert passed is False
    assert summary["min_anomaly_score"] == 0.0
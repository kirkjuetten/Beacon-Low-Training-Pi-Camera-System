from __future__ import annotations

import numpy as np

from scoring_utils import evaluate_metrics, normalize_inspection_mode, resolve_inspection_mode_details, score_sample


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
        "inspection_mode": "full",
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
        "inspection_mode": "full",
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


def test_evaluate_metrics_mask_only_ignores_optional_gates() -> None:
    metrics = {
        "required_coverage": 0.95,
        "outside_allowed_ratio": 0.01,
        "min_section_coverage": 0.9,
        "ssim": 0.1,
        "mse": 999.0,
        "anomaly_score": -5.0,
    }
    inspection_cfg = {
        "inspection_mode": "mask_only",
        "min_required_coverage": 0.92,
        "max_outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.85,
        "min_ssim": 0.9,
        "max_mse": 5.0,
        "min_anomaly_score": 0.0,
    }

    passed, summary = evaluate_metrics(metrics, inspection_cfg)

    assert passed is True
    assert summary["inspection_mode"] == "mask_only"
    assert summary["ssim_gate_active"] is False
    assert summary["mse_gate_active"] is False
    assert summary["anomaly_gate_active"] is False


def test_evaluate_metrics_mask_and_ssim_applies_only_ssim_mse_gates() -> None:
    metrics = {
        "required_coverage": 0.95,
        "outside_allowed_ratio": 0.01,
        "min_section_coverage": 0.9,
        "ssim": 0.95,
        "mse": 3.0,
        "anomaly_score": -5.0,
    }
    inspection_cfg = {
        "inspection_mode": "mask_and_ssim",
        "min_required_coverage": 0.92,
        "max_outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.85,
        "min_ssim": 0.9,
        "max_mse": 5.0,
        "min_anomaly_score": 0.0,
    }

    passed, summary = evaluate_metrics(metrics, inspection_cfg)

    assert passed is True
    assert summary["ssim_gate_active"] is True
    assert summary["mse_gate_active"] is True
    assert summary["anomaly_gate_active"] is False


def test_evaluate_metrics_mask_and_ml_applies_only_anomaly_gate() -> None:
    metrics = {
        "required_coverage": 0.95,
        "outside_allowed_ratio": 0.01,
        "min_section_coverage": 0.9,
        "ssim": 0.1,
        "mse": 999.0,
        "anomaly_score": 0.25,
    }
    inspection_cfg = {
        "inspection_mode": "mask_and_ml",
        "min_required_coverage": 0.92,
        "max_outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.85,
        "min_ssim": 0.9,
        "max_mse": 5.0,
        "min_anomaly_score": 0.0,
    }

    passed, summary = evaluate_metrics(metrics, inspection_cfg)

    assert passed is True
    assert summary["ssim_gate_active"] is False
    assert summary["mse_gate_active"] is False
    assert summary["anomaly_gate_active"] is True


def test_evaluate_metrics_full_only_activates_optional_gates_with_thresholds() -> None:
    metrics = {
        "required_coverage": 0.95,
        "outside_allowed_ratio": 0.01,
        "min_section_coverage": 0.9,
        "ssim": 0.1,
        "mse": 999.0,
        "anomaly_score": 0.25,
    }
    inspection_cfg = {
        "inspection_mode": "full",
        "min_required_coverage": 0.92,
        "max_outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.85,
        "min_anomaly_score": 0.0,
    }

    passed, summary = evaluate_metrics(metrics, inspection_cfg)

    assert passed is True
    assert summary["ssim_gate_active"] is False
    assert summary["mse_gate_active"] is False
    assert summary["anomaly_gate_active"] is True


def test_evaluate_metrics_full_applies_all_optional_gates() -> None:
    metrics = {
        "required_coverage": 0.95,
        "outside_allowed_ratio": 0.01,
        "min_section_coverage": 0.9,
        "ssim": 0.95,
        "mse": 3.0,
        "anomaly_score": -0.2,
    }
    inspection_cfg = {
        "inspection_mode": "full",
        "min_required_coverage": 0.92,
        "max_outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.85,
        "min_ssim": 0.9,
        "max_mse": 5.0,
        "min_anomaly_score": 0.0,
    }

    passed, summary = evaluate_metrics(metrics, inspection_cfg)

    assert passed is False
    assert summary["ssim_gate_active"] is True
    assert summary["mse_gate_active"] is True
    assert summary["anomaly_gate_active"] is True


def test_normalize_inspection_mode_falls_back_to_mask_only() -> None:
    assert normalize_inspection_mode("full") == "full"
    assert normalize_inspection_mode("MASK_AND_ML") == "mask_and_ml"
    assert normalize_inspection_mode("unknown") == "mask_only"


def test_resolve_inspection_mode_details_reports_explicit_gate_combination() -> None:
    details = resolve_inspection_mode_details(
        {
            "inspection_mode": "mask_and_ml",
            "min_ssim": 0.9,
            "max_mse": 5.0,
            "min_anomaly_score": 0.1,
        }
    )

    assert details["inspection_mode"] == "mask_and_ml"
    assert details["included_gates"] == {"anomaly"}
    assert details["ssim_gate_active"] is False
    assert details["mse_gate_active"] is False
    assert details["anomaly_gate_active"] is True
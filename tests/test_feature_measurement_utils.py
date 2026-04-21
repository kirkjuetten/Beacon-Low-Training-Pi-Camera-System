from __future__ import annotations

import cv2
import numpy as np

from inspection_system.app.feature_measurement_utils import extract_molded_part_feature_measurements


def test_extract_molded_part_feature_measurements_reports_isolated_centroid_dx_dy_and_radial_offset() -> None:
    reference_required = np.zeros((80, 120), dtype=np.uint8)
    reference_required[10:60, 10:70] = 255
    reference_required[62:72, 75:85] = 255
    reference_required[62:70, 95:103] = 255

    sample_datum_mask = np.zeros_like(reference_required)
    sample_datum_mask[10:60, 10:70] = 255
    sample_datum_mask[61:71, 78:88] = 255
    sample_datum_mask[62:70, 95:103] = 255

    measurements, summary = extract_molded_part_feature_measurements(
        reference_required,
        sample_datum_mask,
        ["isolated_centroid"],
        cv2,
        np,
    )

    assert [measurement["feature_key"] for measurement in measurements] == ["isolated_feature_1", "isolated_feature_2"]
    assert measurements[0]["feature_family"] == "isolated_centroid"
    assert measurements[0]["feature_type"] == "isolated_centroid_position"
    assert measurements[0]["dx_px"] == 3.0
    assert measurements[0]["dy_px"] == -1.0
    assert round(float(measurements[0]["radial_offset_px"]), 3) == round(float(np.hypot(3.0, -1.0)), 3)
    assert summary is not None
    assert summary["feature_type"] == "isolated_centroid_position"
    assert summary["failure_cause"] == "feature_position"
    assert summary["feature_key"] == "isolated_feature_1"
    assert summary["dx_px"] == 3.0
    assert summary["dy_px"] == -1.0


def test_extract_molded_part_feature_measurements_reports_paired_centroid_spacing() -> None:
    reference_required = np.zeros((80, 120), dtype=np.uint8)
    reference_required[10:60, 10:70] = 255
    reference_required[62:72, 75:85] = 255
    reference_required[62:70, 95:103] = 255

    sample_datum_mask = np.zeros_like(reference_required)
    sample_datum_mask[10:60, 10:70] = 255
    sample_datum_mask[63:73, 77:87] = 255
    sample_datum_mask[63:71, 97:105] = 255

    measurements, summary = extract_molded_part_feature_measurements(
        reference_required,
        sample_datum_mask,
        ["paired_centroid"],
        cv2,
        np,
    )

    assert [measurement["feature_key"] for measurement in measurements] == ["paired_feature_1"]
    assert measurements[0]["feature_family"] == "paired_centroid"
    assert measurements[0]["feature_type"] == "paired_centroid_position"
    assert measurements[0]["dx_px"] == 2.0
    assert measurements[0]["dy_px"] == 1.0
    assert measurements[0]["pair_spacing_delta_px"] == 0.0
    assert summary is not None
    assert summary["feature_key"] == "paired_feature_1"
    assert summary["pair_spacing_delta_px"] == 0.0
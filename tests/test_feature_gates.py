from __future__ import annotations

from inspection_system.app.gates.feature_gates import evaluate_feature_gates


def test_evaluate_feature_gates_fails_when_feature_offset_exceeds_threshold() -> None:
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
            "dy_px": -0.2,
            "radial_offset_px": 1.6124515,
            "center_offset_px": 1.6124515,
        }
    ]

    result = evaluate_feature_gates(
        feature_measurements,
        {"feature_gate_thresholds": {"max_dx_px": 1.0, "max_radial_offset_px": 2.0}},
    )

    assert result["passed"] is False
    assert result["summary"]["feature_gate_active"] is True
    assert result["summary"]["feature_gate_metric"] == "dx_px"
    assert result["summary"]["feature_gate_feature_key"] == "isolated_feature_1"
    assert result["summary"]["feature_gate_margin_px"] < 0.0
    assert result["feature_position_summary"]["feature_key"] == "isolated_feature_1"


def test_evaluate_feature_gates_leaves_datum_section_fallback_inactive() -> None:
    feature_measurements = [
        {
            "feature_key": "section_1",
            "feature_label": "Datum Section 1",
            "feature_family": "datum_section",
            "feature_type": "datum_section_position",
            "measurement_frame": "datum",
            "sample_detected": True,
            "failure_cause": "feature_position",
            "center_offset_px": 2.0,
        }
    ]

    result = evaluate_feature_gates(
        feature_measurements,
        {"feature_gate_thresholds": {"max_dx_px": 1.0, "max_radial_offset_px": 1.0}},
    )

    assert result["passed"] is True
    assert result["summary"]["feature_gate_active"] is False
    assert result["feature_position_summary"] is None
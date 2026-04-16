from inspection_system.app.runtime_controller import get_inspection_runtime_warnings


def test_ml_mode_warns_when_model_and_threshold_are_missing() -> None:
    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "mask_and_ml"}},
        anomaly_detector=None,
    )

    assert len(warnings) == 2
    assert "no trained anomaly model" in warnings[0].lower()
    assert "min anomaly score" in warnings[1].lower()


def test_full_mode_warns_only_for_missing_threshold_when_model_exists() -> None:
    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "full", "min_anomaly_score": None}},
        anomaly_detector=object(),
    )

    assert warnings == [
        "ML-backed mode is selected but Min Anomaly Score is not set. The anomaly gate is inactive."
    ]


def test_mask_only_mode_has_no_ml_warnings() -> None:
    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "mask_only"}},
        anomaly_detector=None,
    )

    assert warnings == []
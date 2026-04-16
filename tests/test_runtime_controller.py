import json

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


def test_ml_mode_warns_when_committed_good_samples_are_insufficient(tmp_path) -> None:
    reference_dir = tmp_path / "reference"
    active_dir = reference_dir / "anomaly_good_library" / "active"
    for index in range(3):
        sample_dir = active_dir / f"sample_{index}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample_meta.json").write_text(
            json.dumps({"sample_asset": {"sample_id": f"sample_{index}"}, "features": [0.1, 0.2, 0.3]}),
            encoding="utf-8",
        )

    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "mask_and_ml", "min_anomaly_score": 0.1}},
        anomaly_detector=None,
        active_paths={"reference_dir": reference_dir},
    )

    assert len(warnings) == 1
    assert "not enough approved-good samples" in warnings[0].lower()
    assert "3/8" in warnings[0]


def test_ml_mode_warns_when_model_is_stale_for_current_sample_library(tmp_path) -> None:
    reference_dir = tmp_path / "reference"
    active_dir = reference_dir / "anomaly_good_library" / "active"
    for index in range(8):
        sample_dir = active_dir / f"sample_{index}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample_meta.json").write_text(
            json.dumps({"sample_asset": {"sample_id": f"sample_{index}"}, "features": [0.1, 0.2, 0.3]}),
            encoding="utf-8",
        )

    (reference_dir / "anomaly_model.pkl").write_bytes(b"model")
    (reference_dir / "anomaly_model_meta.json").write_text(
        json.dumps({"trained_sample_count": 6}),
        encoding="utf-8",
    )

    warnings = get_inspection_runtime_warnings(
        {"inspection": {"inspection_mode": "full", "min_anomaly_score": 0.1}},
        anomaly_detector=object(),
        active_paths={"reference_dir": reference_dir},
    )

    assert warnings == [
        "ML-backed mode is selected but the anomaly model is stale for the current approved-good sample library. Press Update in training to rebuild it."
    ]
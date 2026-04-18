from inspection_system.app.training_labels import default_final_class
from inspection_system.app.training_schema import (
    build_config_fingerprint,
    build_training_record,
    normalize_record_schema,
)


def test_build_config_fingerprint_uses_expected_defaults() -> None:
    fingerprint = build_config_fingerprint({"inspection": {}, "alignment": {}})

    assert fingerprint["inspection_mode"] == "mask_only"
    assert fingerprint["reference_strategy"] == "golden_only"
    assert fingerprint["blend_mode"] == "hard_only"
    assert fingerprint["tolerance_mode"] == "balanced"
    assert fingerprint["alignment_profile"] == "balanced"


def test_normalize_record_schema_backfills_training_fields() -> None:
    record = {"timestamp": 123.0, "feedback": "approve", "metrics": {"required_coverage": 0.95}}
    changed = normalize_record_schema(
        record,
        schema_version=2,
        default_final_class=default_final_class,
        config_fingerprint={"inspection_mode": "mask_only"},
        timestamp_provider=lambda: 999.0,
    )

    assert changed is True
    assert record["schema_version"] == 2
    assert record["final_class"] == "good"
    assert record["config_fingerprint"] == {"inspection_mode": "mask_only"}
    assert record["reference_candidate_id"] is None
    assert record["anomaly_sample_id"] is None


def test_build_training_record_captures_metrics_and_labels() -> None:
    record = build_training_record(
        {
            "required_coverage": 0.91,
            "outside_allowed_ratio": 0.03,
            "min_section_coverage": 0.8,
            "mean_edge_distance_px": 1.2,
            "ssim": 0.88,
            "inspection_mode": "full",
        },
        "review",
        schema_version=2,
        record_id="feedback_1",
        timestamp=12.5,
        final_class="reject",
        label_info={"defect_category": "smear", "classification_reason": "operator"},
        config_fingerprint={"inspection_mode": "full"},
    )

    assert record["record_id"] == "feedback_1"
    assert record["final_class"] == "reject"
    assert record["defect_category"] == "smear"
    assert record["classification_reason"] == "operator"
    assert record["metrics"]["mean_edge_distance_px"] == 1.2
    assert record["metrics"]["ssim"] == 0.88
    assert record["metrics"]["inspection_mode"] == "full"
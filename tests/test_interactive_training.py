import json

from inspection_system.app.interactive_training import ThresholdTrainer


def test_apply_suggestions_updates_config_and_file(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(
        json.dumps(
            {
                "inspection": {
                    "min_required_coverage": 0.92,
                    "max_outside_allowed_ratio": 0.02,
                }
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    trainer = ThresholdTrainer(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    applied = trainer.apply_suggestions(
        config,
        {
            "min_required_coverage": 0.94555,
            "max_outside_allowed_ratio": 0.01555,
        },
    )

    assert applied == {
        "min_required_coverage": 0.9456,
        "max_outside_allowed_ratio": 0.0155,
    }
    assert config["inspection"]["min_required_coverage"] == 0.9456
    assert config["inspection"]["max_outside_allowed_ratio"] == 0.0155

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["inspection"]["min_required_coverage"] == 0.9456
    assert saved_config["inspection"]["max_outside_allowed_ratio"] == 0.0155


def test_apply_suggestions_skips_unchanged_values(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(
        json.dumps(
            {
                "inspection": {
                    "min_required_coverage": 0.9455,
                }
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    trainer = ThresholdTrainer(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    applied = trainer.apply_suggestions(config, {"min_required_coverage": 0.9455})

    assert applied == {}


def test_pending_feedback_summary_and_lifecycle(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(json.dumps({"inspection": {}}, indent=2) + "\n", encoding="utf-8")

    trainer = ThresholdTrainer(config_path)
    details = {
        "required_coverage": 0.95,
        "outside_allowed_ratio": 0.01,
        "min_section_coverage": 0.9,
    }

    trainer.record_feedback(details, "approve")
    trainer.record_feedback(details, "reject")
    trainer.record_feedback(details, "review")

    summary = trainer.get_pending_summary()
    assert summary == {"approve": 1, "reject": 1, "review": 1, "total": 3}

    committed = trainer.commit_pending_feedback()
    assert committed == 3
    summary_after_commit = trainer.get_pending_summary()
    assert summary_after_commit == {"approve": 0, "reject": 0, "review": 0, "total": 0}

    saved_training = json.loads((config_path.parent / "training_data.json").read_text(encoding="utf-8"))
    assert saved_training[0]["schema_version"] == 2
    assert saved_training[0]["final_class"] == "good"
    assert saved_training[1]["final_class"] == "reject"
    assert saved_training[2]["final_class"] is None
    assert "config_fingerprint" in saved_training[0]
    assert saved_training[0]["metrics"]["mse"] is None


def test_discard_pending_feedback_marks_records_without_deleting(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(json.dumps({"inspection": {}}, indent=2) + "\n", encoding="utf-8")

    trainer = ThresholdTrainer(config_path)
    details = {
        "required_coverage": 0.92,
        "outside_allowed_ratio": 0.02,
        "min_section_coverage": 0.85,
    }
    trainer.record_feedback(details, "approve")
    trainer.record_feedback(details, "reject")

    discarded = trainer.discard_pending_feedback()
    assert discarded == 2

    saved_training = json.loads((config_path.parent / "training_data.json").read_text(encoding="utf-8"))
    assert len(saved_training) == 2
    assert all(record.get("learning_state") == "discarded" for record in saved_training)


def test_record_feedback_accepts_final_class_and_defect_category(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(json.dumps({"inspection": {"inspection_mode": "full"}}, indent=2) + "\n", encoding="utf-8")

    trainer = ThresholdTrainer(config_path)
    trainer.record_feedback(
        {
            "required_coverage": 0.88,
            "outside_allowed_ratio": 0.04,
            "min_section_coverage": 0.72,
            "ssim": 0.81,
            "mse": 7.2,
            "anomaly_score": -0.5,
            "histogram_similarity": 0.33,
            "best_angle_deg": 0.4,
            "best_shift_x": 3,
            "best_shift_y": -1,
            "inspection_mode": "full",
        },
        "review",
        {
            "final_class": "reject",
            "defect_category": "smear",
            "classification_reason": "trainer_override",
        },
    )

    saved_training = json.loads((config_path.parent / "training_data.json").read_text(encoding="utf-8"))
    record = saved_training[0]
    assert record["feedback"] == "review"
    assert record["final_class"] == "reject"
    assert record["defect_category"] == "smear"
    assert record["classification_reason"] == "trainer_override"
    assert record["metrics"]["mse"] == 7.2
    assert record["metrics"]["histogram_similarity"] == 0.33
    assert record["metrics"]["best_shift_x"] == 3


def test_load_training_data_backfills_schema_for_legacy_records(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(json.dumps({"inspection": {}}, indent=2) + "\n", encoding="utf-8")
    training_path = config_path.parent / "training_data.json"
    training_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": 123.0,
                    "feedback": "approve",
                    "metrics": {"required_coverage": 0.95},
                },
                {
                    "timestamp": 124.0,
                    "feedback": "reject",
                    "learning_state": "pending",
                    "metrics": {"required_coverage": 0.75},
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    trainer = ThresholdTrainer(config_path)

    assert trainer.training_data[0]["learning_state"] == "committed"
    assert trainer.training_data[0]["schema_version"] == 2
    assert trainer.training_data[0]["final_class"] == "good"
    assert trainer.training_data[1]["final_class"] == "reject"
    assert trainer.training_data[1]["defect_category"] is None


def test_training_review_warnings_surface_config_fit_problems(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config = {
        "inspection": {
            "inspection_mode": "mask_only",
            "reference_strategy": "golden_only",
            "blend_mode": "hard_only",
            "tolerance_mode": "balanced",
            "min_required_coverage": 0.92,
            "max_outside_allowed_ratio": 0.02,
            "min_section_coverage": 0.85,
        },
        "alignment": {"tolerance_profile": "balanced"},
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    trainer = ThresholdTrainer(config_path)
    trainer.record_feedback(
        {"required_coverage": 0.80, "outside_allowed_ratio": 0.01, "min_section_coverage": 0.90},
        "approve",
    )
    trainer.record_feedback(
        {"required_coverage": 0.99, "outside_allowed_ratio": 0.0, "min_section_coverage": 0.99},
        "reject",
    )

    changed_config = json.loads(json.dumps(config))
    changed_config["inspection"]["tolerance_mode"] = "strict"

    warnings = trainer.get_training_review_warnings(
        changed_config,
        runtime_warnings=["ML-backed mode is selected but no trained anomaly model is available."],
        reference_warning="Threshold value mismatch: reference was 180.0; current is 170.0",
    )

    assert any("no trained anomaly model" in warning.lower() for warning in warnings)
    assert any("threshold value mismatch" in warning.lower() for warning in warnings)
    assert any("different config settings" in warning.lower() for warning in warnings)
    assert any("approved-good pending examples fail" in warning.lower() for warning in warnings)
    assert any("reject-labeled pending examples still pass" in warning.lower() for warning in warnings)
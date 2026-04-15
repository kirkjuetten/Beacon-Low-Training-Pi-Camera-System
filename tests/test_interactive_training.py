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
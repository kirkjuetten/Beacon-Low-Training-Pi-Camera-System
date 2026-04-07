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
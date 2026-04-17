import json

import inspection_system.app.interactive_training as interactive_training
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


def test_extract_learned_ranges_collects_good_and_reject_statistics(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(json.dumps({"inspection": {}}, indent=2) + "\n", encoding="utf-8")

    trainer = ThresholdTrainer(config_path)
    trainer.record_feedback(
        {"required_coverage": 0.82, "outside_allowed_ratio": 0.03, "min_section_coverage": 0.70, "mean_edge_distance_px": 1.2, "worst_section_edge_distance_px": 1.4, "worst_section_width_delta_ratio": 0.18, "worst_section_center_offset_px": 0.9, "ssim": 0.88, "mse": 7.0},
        "approve",
    )
    trainer.record_feedback(
        {"required_coverage": 0.90, "outside_allowed_ratio": 0.01, "min_section_coverage": 0.88, "mean_edge_distance_px": 0.5, "worst_section_edge_distance_px": 0.7, "worst_section_width_delta_ratio": 0.08, "worst_section_center_offset_px": 0.3, "ssim": 0.94, "mse": 3.0},
        "approve",
    )
    trainer.record_feedback(
        {"required_coverage": 0.76, "outside_allowed_ratio": 0.05, "min_section_coverage": 0.60, "mean_edge_distance_px": 2.0, "worst_section_edge_distance_px": 2.3, "worst_section_width_delta_ratio": 0.28, "worst_section_center_offset_px": 1.8, "ssim": 0.80, "mse": 9.0},
        "reject",
    )

    learned_ranges = trainer.extract_learned_ranges()

    assert learned_ranges["required_coverage"]["good_min"] == 0.82
    assert learned_ranges["required_coverage"]["good_max"] == 0.9
    assert learned_ranges["required_coverage"]["reject_max"] == 0.76
    assert learned_ranges["outside_allowed_ratio"]["good_max"] == 0.03
    assert learned_ranges["mean_edge_distance_px"]["direction"] == "lower_is_better"
    assert learned_ranges["mean_edge_distance_px"]["good_max"] == 1.2
    assert learned_ranges["worst_section_edge_distance_px"]["direction"] == "lower_is_better"
    assert learned_ranges["worst_section_edge_distance_px"]["good_max"] == 1.4
    assert learned_ranges["worst_section_width_delta_ratio"]["direction"] == "lower_is_better"
    assert learned_ranges["worst_section_width_delta_ratio"]["good_max"] == 0.18
    assert learned_ranges["worst_section_center_offset_px"]["direction"] == "lower_is_better"
    assert learned_ranges["worst_section_center_offset_px"]["good_max"] == 0.9
    assert learned_ranges["mse"]["direction"] == "lower_is_better"


def test_apply_learning_update_persists_learned_ranges(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(json.dumps({"inspection": {"blend_mode": "blend_balanced"}}, indent=2) + "\n", encoding="utf-8")

    trainer = ThresholdTrainer(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    learned_ranges = {
        "required_coverage": {
            "direction": "higher_is_better",
            "good_min": 0.82,
            "good_max": 0.95,
            "good_mean": 0.9,
            "good_count": 12,
        }
    }

    applied = trainer.apply_learning_update(
        config,
        {"min_required_coverage": 0.82},
        learned_ranges,
    )

    assert applied["threshold_updates"] == {"min_required_coverage": 0.82}
    assert applied["learned_ranges_saved"] is True
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config["inspection"]["learned_ranges"] == learned_ranges


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
            "mean_edge_distance_px": 1.9,
            "worst_section_edge_distance_px": 2.1,
            "worst_section_width_delta_ratio": 0.22,
            "worst_section_center_offset_px": 1.1,
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
    assert record["metrics"]["mean_edge_distance_px"] == 1.9
    assert record["metrics"]["worst_section_edge_distance_px"] == 2.1
    assert record["metrics"]["worst_section_width_delta_ratio"] == 0.22
    assert record["metrics"]["worst_section_center_offset_px"] == 1.1
    assert record["metrics"]["histogram_similarity"] == 0.33
    assert record["metrics"]["best_shift_x"] == 3


def test_suggest_thresholds_can_generate_edge_distance_limit(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(
        json.dumps({"inspection": {"max_mean_edge_distance_px": 2.0, "max_section_edge_distance_px": 2.5, "max_section_width_delta_ratio": 0.4, "max_section_center_offset_px": 2.0}}, indent=2) + "\n",
        encoding="utf-8",
    )

    trainer = ThresholdTrainer(config_path)
    trainer.training_data = []
    for index, edge_distance in enumerate([0.4, 0.5, 0.6, 0.55, 0.45, 0.5, 0.52, 0.48, 0.51, 0.53], start=1):
        trainer.training_data.append(
            {
                "schema_version": 2,
                "record_id": f"feedback_{index}",
                "timestamp": float(index),
                "feedback": "approve",
                "final_class": "good",
                "defect_category": None,
                "classification_reason": None,
                "learning_state": "pending",
                "config_fingerprint": trainer._build_config_fingerprint(),
                "reference_candidate_id": None,
                "reference_candidate_state": None,
                "anomaly_sample_id": None,
                "anomaly_sample_state": None,
                "metrics": {
                    "required_coverage": 0.95,
                    "outside_allowed_ratio": 0.01,
                    "min_section_coverage": 0.9,
                    "mean_edge_distance_px": edge_distance,
                    "worst_section_edge_distance_px": edge_distance + 0.2,
                    "worst_section_width_delta_ratio": round(edge_distance / 10.0, 3),
                    "worst_section_center_offset_px": round(edge_distance + 0.1, 3),
                },
            }
        )

    suggestions = trainer.suggest_thresholds()

    assert suggestions["max_mean_edge_distance_px"] == 0.6
    assert suggestions["max_section_edge_distance_px"] == 0.8
    assert suggestions["max_section_width_delta_ratio"] == 0.06
    assert suggestions["max_section_center_offset_px"] == 0.7


def test_suggest_thresholds_can_loosen_lower_is_better_geometry_thresholds(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(
        json.dumps(
            {
                "inspection": {
                    "max_mean_edge_distance_px": 0.35,
                    "max_section_edge_distance_px": 0.45,
                    "max_section_width_delta_ratio": 0.03,
                    "max_section_center_offset_px": 0.25,
                }
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    trainer = ThresholdTrainer(config_path)
    trainer.training_data = []
    for index, edge_distance in enumerate([0.4, 0.5, 0.6, 0.55, 0.45], start=1):
        trainer.training_data.append(
            {
                "schema_version": 2,
                "record_id": f"feedback_{index}",
                "timestamp": float(index),
                "feedback": "approve",
                "final_class": "good",
                "defect_category": None,
                "classification_reason": None,
                "learning_state": "pending",
                "config_fingerprint": trainer._build_config_fingerprint(),
                "reference_candidate_id": None,
                "reference_candidate_state": None,
                "anomaly_sample_id": None,
                "anomaly_sample_state": None,
                "metrics": {
                    "required_coverage": 0.95,
                    "outside_allowed_ratio": 0.01,
                    "min_section_coverage": 0.9,
                    "mean_edge_distance_px": edge_distance,
                    "worst_section_edge_distance_px": edge_distance + 0.2,
                    "worst_section_width_delta_ratio": round(edge_distance / 10.0, 3),
                    "worst_section_center_offset_px": round(edge_distance + 0.1, 3),
                },
            }
        )

    suggestions = trainer.suggest_thresholds()

    assert suggestions["max_mean_edge_distance_px"] == 0.6
    assert suggestions["max_section_edge_distance_px"] == 0.8
    assert suggestions["max_section_width_delta_ratio"] == 0.06
    assert suggestions["max_section_center_offset_px"] == 0.7


def test_suggest_thresholds_uses_committed_learning_records(tmp_path) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(
        json.dumps({"inspection": {"max_section_center_offset_px": 0.2}}, indent=2) + "\n",
        encoding="utf-8",
    )

    trainer = ThresholdTrainer(config_path)
    trainer.training_data = []
    for index, center_offset in enumerate([0.3, 0.35, 0.4, 0.45, 0.5], start=1):
        trainer.training_data.append(
            {
                "schema_version": 2,
                "record_id": f"feedback_{index}",
                "timestamp": float(index),
                "feedback": "approve",
                "final_class": "good",
                "defect_category": None,
                "classification_reason": None,
                "learning_state": "committed",
                "config_fingerprint": trainer._build_config_fingerprint(),
                "reference_candidate_id": None,
                "reference_candidate_state": None,
                "anomaly_sample_id": None,
                "anomaly_sample_state": None,
                "metrics": {
                    "required_coverage": 0.95,
                    "outside_allowed_ratio": 0.01,
                    "min_section_coverage": 0.9,
                    "worst_section_center_offset_px": center_offset,
                },
            }
        )

    suggestions = trainer.suggest_thresholds()

    assert suggestions["max_section_center_offset_px"] == 0.5


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


def test_record_feedback_stages_candidate_for_approved_good_when_multi_reference_enabled(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(json.dumps({"inspection": {"reference_strategy": "hybrid"}}, indent=2) + "\n", encoding="utf-8")

    trainer = ThresholdTrainer(config_path)
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"img")
    captured = {}
    anomaly_captured = {}

    def fake_stage_reference_candidate_from_image(config, source_image_path, **kwargs):
        captured['image_path'] = source_image_path
        return True, {'reference_id': 'candidate_123', 'state': 'pending'}

    def fake_stage_anomaly_training_sample_from_image(config, source_image_path, **kwargs):
        anomaly_captured['image_path'] = source_image_path
        return True, {'sample_id': 'sample_123', 'state': 'pending'}

    monkeypatch.setattr(interactive_training, 'stage_reference_candidate_from_image', fake_stage_reference_candidate_from_image)
    monkeypatch.setattr(interactive_training, 'stage_anomaly_training_sample_from_image', fake_stage_anomaly_training_sample_from_image)

    trainer.record_feedback(
        {
            "required_coverage": 0.95,
            "outside_allowed_ratio": 0.01,
            "min_section_coverage": 0.9,
        },
        "approve",
        image_path=image_path,
    )

    assert captured['image_path'] == image_path
    assert anomaly_captured['image_path'] == image_path
    saved_training = json.loads((config_path.parent / "training_data.json").read_text(encoding="utf-8"))
    assert saved_training[0]["reference_candidate_id"] == "candidate_123"
    assert saved_training[0]["reference_candidate_state"] == "pending"
    assert saved_training[0]["anomaly_sample_id"] == "sample_123"
    assert saved_training[0]["anomaly_sample_state"] == "pending"


def test_commit_and_discard_pending_feedback_manage_candidate_reference_state(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(json.dumps({"inspection": {}}, indent=2) + "\n", encoding="utf-8")

    trainer = ThresholdTrainer(config_path)
    trainer.training_data = [
        {
            'schema_version': 2,
            'record_id': 'feedback_1',
            'timestamp': 1.0,
            'feedback': 'approve',
            'final_class': 'good',
            'defect_category': None,
            'classification_reason': None,
            'learning_state': 'pending',
            'config_fingerprint': {},
            'reference_candidate_id': 'candidate_1',
            'reference_candidate_state': 'pending',
            'anomaly_sample_id': 'sample_1',
            'anomaly_sample_state': 'pending',
            'metrics': {'required_coverage': 0.95},
        },
        {
            'schema_version': 2,
            'record_id': 'feedback_2',
            'timestamp': 2.0,
            'feedback': 'approve',
            'final_class': 'good',
            'defect_category': None,
            'classification_reason': None,
            'learning_state': 'pending',
            'config_fingerprint': {},
            'reference_candidate_id': 'candidate_2',
            'reference_candidate_state': 'pending',
            'anomaly_sample_id': 'sample_2',
            'anomaly_sample_state': 'pending',
            'metrics': {'required_coverage': 0.94},
        },
    ]

    activated = []
    discarded = []
    activated_samples = []
    discarded_samples = []
    monkeypatch.setattr(interactive_training, 'activate_reference_candidate', lambda candidate_id: activated.append(candidate_id) or True)
    monkeypatch.setattr(interactive_training, 'discard_reference_candidate', lambda candidate_id, state='pending': discarded.append((candidate_id, state)) or True)
    monkeypatch.setattr(interactive_training, 'activate_anomaly_training_sample', lambda sample_id: activated_samples.append(sample_id) or True)
    monkeypatch.setattr(interactive_training, 'discard_anomaly_training_sample', lambda sample_id, state='pending': discarded_samples.append((sample_id, state)) or True)

    committed = trainer.commit_pending_feedback()
    assert committed == 2
    assert activated == ['candidate_1', 'candidate_2']
    assert activated_samples == ['sample_1', 'sample_2']
    assert all(record['reference_candidate_state'] == 'active' for record in trainer.training_data)
    assert all(record['anomaly_sample_state'] == 'active' for record in trainer.training_data)

    for record in trainer.training_data:
        record['learning_state'] = 'pending'
        record['reference_candidate_state'] = 'pending'
        record['anomaly_sample_state'] = 'pending'

    discarded_count = trainer.discard_pending_feedback()
    assert discarded_count == 2
    assert discarded == [('candidate_1', 'pending'), ('candidate_2', 'pending')]
    assert discarded_samples == [('sample_1', 'pending'), ('sample_2', 'pending')]
    assert all(record['reference_candidate_state'] == 'discarded' for record in trainer.training_data)
    assert all(record['anomaly_sample_state'] == 'discarded' for record in trainer.training_data)
    assert trainer.training_data[1]["defect_category"] is None


def test_rebuild_anomaly_model_delegates_to_reference_service(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "camera_config.json"
    config_path.write_text(json.dumps({"inspection": {}}, indent=2) + "\n", encoding="utf-8")

    trainer = ThresholdTrainer(config_path)
    called = {}

    def fake_train_anomaly_model_from_samples(config):
        called['config'] = config
        return {'rebuilt': True, 'trained_sample_count': 8}

    monkeypatch.setattr(interactive_training, 'train_anomaly_model_from_samples', fake_train_anomaly_model_from_samples)

    result = trainer.rebuild_anomaly_model({'inspection': {'inspection_mode': 'mask_and_ml'}})

    assert called['config']['inspection']['inspection_mode'] == 'mask_and_ml'
    assert result['rebuilt'] is True


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
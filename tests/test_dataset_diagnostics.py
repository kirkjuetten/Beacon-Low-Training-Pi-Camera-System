import json
from pathlib import Path

from inspection_system.app.dataset_diagnostics import (
    FAIL,
    INVALID_CAPTURE,
    PASS,
    REGISTRATION_FAILED,
    _diagnose_result,
    build_active_paths_from_project_root,
    build_diagnostic_output_path,
    duplicate_training_records,
    expected_status_for_record,
    load_capture_records,
    partition_episode_records,
    resolve_project_context,
    resolve_capture_manifests,
    resolve_project_root_from_source,
    save_episode_report,
    simulate_training_episode,
)


def _write_session(tmp_path: Path) -> Path:
    session_dir = tmp_path / "test_data" / "5096" / "angle-a" / "session_a"
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"bucket": "good", "dataset_split": "tuning", "image_path": str(images_dir / "good_a.png"), "expected_inspection_status": PASS},
        {"bucket": "good", "dataset_split": "tuning", "image_path": str(images_dir / "good_b.png"), "expected_inspection_status": PASS},
        {"bucket": "reject", "dataset_split": "tuning", "image_path": str(images_dir / "reject_a.png"), "expected_inspection_status": FAIL, "defect_category": "broken_coring"},
        {"bucket": "good", "dataset_split": "validation", "image_path": str(images_dir / "good_eval.png"), "expected_inspection_status": PASS},
        {"bucket": "reject", "dataset_split": "validation", "image_path": str(images_dir / "reject_eval.png"), "expected_inspection_status": FAIL, "defect_category": "broken_coring"},
        {"bucket": "invalid_capture", "dataset_split": "validation", "image_path": str(images_dir / "invalid_eval.png"), "expected_inspection_status": INVALID_CAPTURE},
        {"bucket": "borderline", "dataset_split": "validation", "image_path": str(images_dir / "borderline_eval.png"), "expected_inspection_status": None},
    ]
    for record in records:
        Path(record["image_path"]).write_bytes(b"img")
    manifest_path = session_dir / "captures.jsonl"
    manifest_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return session_dir


def test_resolve_and_load_capture_records(tmp_path: Path) -> None:
    session_dir = _write_session(tmp_path)

    manifests = resolve_capture_manifests(tmp_path / "test_data")
    records = load_capture_records(session_dir)

    assert manifests == [session_dir / "captures.jsonl"]
    assert len(records) == 7
    assert records[0]["manifest_path"].endswith("captures.jsonl")


def test_load_capture_records_rebases_missing_absolute_image_paths_to_local_session(tmp_path: Path) -> None:
    session_dir = tmp_path / "test_data" / "5096" / "angle-a" / "session_a"
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    local_image = images_dir / "0001_good.png"
    local_image.write_bytes(b"img")
    manifest_path = session_dir / "captures.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "bucket": "good",
                "dataset_split": "tuning",
                "image_path": "/home/pi/inspection_system/projects/5096v2.0/test_data/5096/angle-a/session_a/images/0001_good.png",
                "relative_image_path": "images/0001_good.png",
                "expected_inspection_status": PASS,
            }
        ) + "\n",
        encoding="utf-8",
    )

    records = load_capture_records(session_dir)

    assert records[0]["image_path"] == str(local_image)


def test_partition_episode_records_falls_back_when_no_eval_split_exists() -> None:
    records = [
        {"bucket": "good", "dataset_split": "tuning", "expected_inspection_status": PASS},
        {"bucket": "reject", "dataset_split": "tuning", "expected_inspection_status": FAIL},
    ]

    training, evaluation, info = partition_episode_records(records, eval_splits=("validation",))

    assert len(training) == 2
    assert len(evaluation) == 2
    assert info["evaluation_reused_training_pool"] is True


def test_duplicate_training_records_expands_and_shuffles() -> None:
    records = [{"capture_id": "a"}, {"capture_id": "b"}]

    duplicated = duplicate_training_records(records, 2, shuffle_seed=1)

    assert len(duplicated) == 4
    assert sorted(record["capture_id"] for record in duplicated) == ["a", "a", "b", "b"]


def test_expected_status_for_record_uses_bucket_fallback() -> None:
    assert expected_status_for_record({"bucket": "good", "expected_inspection_status": None}) == PASS
    assert expected_status_for_record({"bucket": "reject"}) == FAIL


def test_diagnose_result_distinguishes_registration_failure() -> None:
    diagnosis = _diagnose_result(
        {},
        {
            "status": REGISTRATION_FAILED,
            "registration_rejection_reason": "Registration confidence 0.500 was below required minimum 0.900.",
            "registration_quality_gate_failures": [
                {
                    "cause_code": "registration_failure",
                    "gate_key": "min_confidence",
                    "summary": "Registration confidence 0.500 was below required minimum 0.900.",
                }
            ],
        },
    )

    assert diagnosis["primary_cause"] == "registration_failure"
    assert diagnosis["failure_modes"][0]["gate_key"] == "min_confidence"
    assert "Registration confidence" in diagnosis["summary"]


def test_diagnose_result_groups_named_molded_feature_position_failures() -> None:
    diagnosis = _diagnose_result(
        {},
        {
            "status": FAIL,
            "inspection_failure_cause": "feature_position",
            "feature_position_summary": {
                "feature_key": "paired_feature_1",
                "feature_label": "Paired Feature 1",
                "feature_family": "paired_centroid",
                "feature_type": "paired_centroid_position",
                "sample_detected": True,
                "dx_px": 3.0,
                "dy_px": -1.0,
                "radial_offset_px": 3.162278,
                "center_offset_px": 3.0,
                "pair_spacing_delta_px": 0.75,
            },
            "best_angle_deg": 0.0,
            "best_shift_x": 0,
            "best_shift_y": 0,
            "max_section_center_offset_px": 1.0,
            "effective_max_section_center_offset_px": 1.0,
            "section_center_gate_active": True,
        },
    )

    assert diagnosis["primary_cause"] == "feature_position"
    assert diagnosis["failure_modes"][0]["cause_code"] == "feature_position"
    assert "Paired Feature 1" in diagnosis["summary"]
    assert "spacing delta 0.75px" in diagnosis["summary"]


def test_resolve_project_context_uses_project_snapshot_near_source(tmp_path: Path) -> None:
    project_root = tmp_path / "inspection_system" / "projects" / "5096v2.0"
    config_file = project_root / "config" / "camera_config.json"
    reference_dir = project_root / "reference"
    test_data_dir = project_root / "test_data" / "5096" / "angle-a" / "session_a"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    test_data_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps({"inspection": {"inspection_mode": "mask_only"}}, indent=2) + "\n", encoding="utf-8")
    (reference_dir / "golden_reference_mask.png").write_bytes(b"mask")
    (reference_dir / "golden_reference_image.png").write_bytes(b"image")

    resolved_root = resolve_project_root_from_source(test_data_dir)
    active_paths = build_active_paths_from_project_root(project_root)
    config, resolved_active_paths, project_context = resolve_project_context(test_data_dir)

    assert resolved_root == project_root
    assert resolved_active_paths == active_paths
    assert project_context == project_root
    assert config["inspection"]["inspection_mode"] == "mask_only"


def test_simulate_training_episode_reports_confusion_metrics(tmp_path: Path, monkeypatch) -> None:
    session_dir = _write_session(tmp_path)
    project_root = tmp_path / "project_5096"
    config_file = project_root / "config" / "camera_config.json"
    reference_dir = project_root / "reference"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / "golden_reference_mask.png").write_bytes(b"mask")
    (reference_dir / "golden_reference_image.png").write_bytes(b"image")
    config_file.write_text(
        json.dumps(
            {
                "inspection": {
                    "inspection_mode": "mask_only",
                    "reference_strategy": "golden_only",
                    "min_required_coverage": 0.9,
                    "max_outside_allowed_ratio": 0.02,
                    "min_section_coverage": 0.85,
                },
                "capture": {},
                "alignment": {"tolerance_profile": "balanced"},
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    active_paths = {
        "config_file": config_file,
        "reference_dir": reference_dir,
        "reference_mask": reference_dir / "golden_reference_mask.png",
        "reference_image": reference_dir / "golden_reference_image.png",
        "log_dir": project_root / "logs",
    }

    def fake_inspect_file(_config, image_path: Path, active_paths=None):
        name = image_path.name
        if name == "good_eval.png":
            status = FAIL
        elif name == "reject_eval.png":
            status = PASS
        elif name == "invalid_eval.png":
            status = PASS
        elif name == "borderline_eval.png":
            status = FAIL
        elif "reject" in name:
            status = FAIL
        else:
            status = PASS
        required_coverage = 0.96 if status == PASS else 0.72
        outside_allowed_ratio = 0.01 if status == PASS else 0.08
        min_section_coverage = 0.92 if status == PASS else 0.84
        if name == "reject_eval.png":
            required_coverage = 0.99
            outside_allowed_ratio = 0.0
            min_section_coverage = 0.97
        return {
            "image": str(image_path),
            "status": status,
            "required_coverage": required_coverage,
            "outside_allowed_ratio": outside_allowed_ratio,
            "min_section_coverage": min_section_coverage,
            "min_required_coverage": 0.9,
            "max_outside_allowed_ratio": 0.02,
            "min_section_coverage_limit": 0.85,
            "effective_min_required_coverage": 0.9,
            "effective_max_outside_allowed_ratio": 0.02,
            "effective_min_section_coverage": 0.85,
            "best_angle_deg": 0.0,
            "best_shift_x": 0,
            "best_shift_y": 0,
            "inspection_mode": "mask_only",
        }

    monkeypatch.setattr("inspection_system.app.dataset_diagnostics.inspect_file", fake_inspect_file)
    monkeypatch.setattr("inspection_system.app.interactive_training.stage_reference_candidate_from_image", lambda *args, **kwargs: (False, "skip"))
    monkeypatch.setattr("inspection_system.app.interactive_training.stage_anomaly_training_sample_from_image", lambda *args, **kwargs: (False, "skip"))

    report = simulate_training_episode(
        session_dir,
        config=json.loads(config_file.read_text(encoding="utf-8")),
        active_paths=active_paths,
        duplicate_count=2,
        update_every=2,
        shuffle_seed=7,
    )

    assert report["training"]["processed_count"] == 6
    assert report["evaluation"]["false_reject_count"] == 1
    assert report["evaluation"]["false_accept_count"] == 1
    assert report["evaluation"]["invalid_capture_miss_count"] == 1
    assert report["evaluation"]["borderline_outcomes"][FAIL] == 1
    assert report["analysis"]["false_reject_patterns"][0]["cause_code"] == "required_coverage"
    assert report["analysis"]["false_accept_patterns"]["feature_gap_categories"]["broken_coring"] == 1
    assert report["analysis"]["recommendations"]
    assert report["analysis"]["recommendations"][0]["title"]
    assert report["training"]["update_events"]
    assert report["training"]["commissioning_status"]["ready"] is False


def test_save_episode_report_writes_json(tmp_path: Path) -> None:
    output_path = build_diagnostic_output_path(tmp_path)
    report_path = save_episode_report({"ok": True}, output_path)

    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"ok": True}
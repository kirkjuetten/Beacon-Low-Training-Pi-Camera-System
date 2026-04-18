import json
from pathlib import Path

from inspection_system.app.dataset_capture import (
    FAIL,
    PASS,
    build_collection_defaults,
    capture_dataset_sample,
    create_capture_session,
    expected_runtime_status,
    format_display_path,
    get_test_data_root,
    persist_collection_defaults,
    should_use_compact_capture_layout,
    slugify_identifier,
)


def test_slugify_identifier_normalizes_values() -> None:
    assert slugify_identifier("5096 Angle A") == "5096-angle-a"
    assert slugify_identifier("", fallback="fallback") == "fallback"


def test_format_display_path_shortens_long_paths() -> None:
    assert format_display_path(Path("/home/pi/inspection_system/projects/5096v2.0/config/camera_config.json"), keep_parts=3) == (
        ".../5096v2.0/config/camera_config.json"
    )


def test_should_use_compact_capture_layout_detects_small_screens() -> None:
    assert should_use_compact_capture_layout(800, 480) is True
    assert should_use_compact_capture_layout(1280, 800) is False


def test_build_collection_defaults_prefills_project_and_setup(tmp_path) -> None:
    active_paths = {
        "config_file": tmp_path / "config" / "camera_config.json",
        "reference_dir": tmp_path / "reference",
    }
    active_paths["config_file"].parent.mkdir(parents=True, exist_ok=True)
    active_paths["reference_dir"].mkdir(parents=True, exist_ok=True)

    defaults = build_collection_defaults(
        {
            "inspection": {"inspection_mode": "full", "reference_strategy": "hybrid", "roi": {"x": 2, "y": 3}},
            "capture": {"timeout_ms": 250},
            "dataset_capture": {"part_id": "5096", "camera_setup_id": "angle_a", "default_split": "regression"},
        },
        active_paths,
        project_name="beacon-5096",
    )

    assert defaults["project_id"] == "beacon-5096"
    assert defaults["part_id"] == "5096"
    assert defaults["camera_setup_id"] == "angle_a"
    assert defaults["default_split"] == "regression"
    assert defaults["inspection_mode"] == "full"
    assert defaults["reference_strategy"] == "hybrid"
    assert defaults["roi_snapshot"] == {"x": 2, "y": 3}


def test_persist_collection_defaults_updates_config_file(tmp_path) -> None:
    config_file = tmp_path / "camera_config.json"
    config = {"inspection": {}, "capture": {}}

    persist_collection_defaults(
        config_file,
        config,
        part_id="5096",
        camera_setup_id="angle_a",
        default_split="validation",
        auto_replay_after_capture=False,
    )

    saved = json.loads(config_file.read_text(encoding="utf-8"))
    assert saved["dataset_capture"] == {
        "part_id": "5096",
        "camera_setup_id": "angle_a",
        "default_split": "validation",
        "auto_replay_after_capture": False,
    }


def test_create_capture_session_creates_project_scoped_session_files(tmp_path) -> None:
    project_root = tmp_path / "project_5096"
    config_file = project_root / "config" / "camera_config.json"
    reference_dir = project_root / "reference"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    active_paths = {"config_file": config_file, "reference_dir": reference_dir}
    session = create_capture_session(
        {"inspection": {}, "capture": {}},
        active_paths,
        "project_5096",
        {
            "part_id": "5096",
            "camera_setup_id": "angle_a",
            "dataset_split": "tuning",
            "session_label": "commissioning",
            "auto_replay_after_capture": True,
        },
    )

    assert get_test_data_root(active_paths) == project_root / "test_data"
    assert session.images_dir.exists()
    metadata = json.loads(session.session_metadata_path.read_text(encoding="utf-8"))
    assert metadata["part_id"] == "5096"
    assert metadata["camera_setup_id"] == "angle_a"
    assert metadata["dataset_split"] == "tuning"


def test_capture_dataset_sample_records_manifest_and_replay_result(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project_5096"
    config_file = project_root / "config" / "camera_config.json"
    reference_dir = project_root / "reference"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    active_paths = {"config_file": config_file, "reference_dir": reference_dir}

    session = create_capture_session(
        {"inspection": {"inspection_mode": "mask_only", "reference_strategy": "golden_only"}, "capture": {}},
        active_paths,
        "project_5096",
        {
            "part_id": "5096",
            "camera_setup_id": "angle_a",
            "dataset_split": "regression",
            "session_label": "baseline",
            "auto_replay_after_capture": True,
        },
    )

    temp_capture = tmp_path / "temp_capture.png"
    temp_capture.write_bytes(b"image-bytes")

    monkeypatch.setattr(
        "inspection_system.app.dataset_capture.capture_to_temp",
        lambda _config: (0, temp_capture, ""),
    )
    monkeypatch.setattr(
        "inspection_system.app.dataset_capture.inspect_file",
        lambda _config, image_path: {"status": FAIL, "image": str(image_path)},
    )

    record = capture_dataset_sample(
        {"inspection": {"inspection_mode": "mask_only", "reference_strategy": "golden_only"}},
        session,
        bucket="reject",
        defect_category="light_pipe_position",
        note="Known feature shift",
    )

    assert Path(record["image_path"]).exists()
    assert record["expected_inspection_status"] == FAIL
    assert record["actual_inspection_status"] == FAIL
    assert record["result_mismatch"] is False
    assert record["defect_category"] == "light_pipe_position"

    lines = session.manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    saved_record = json.loads(lines[0])
    assert saved_record["note"] == "Known feature shift"


def test_expected_runtime_status_matches_supported_buckets() -> None:
    assert expected_runtime_status("good") == PASS
    assert expected_runtime_status("reject") == FAIL
    assert expected_runtime_status("invalid_capture") == "INVALID_CAPTURE"
    assert expected_runtime_status("borderline") is None
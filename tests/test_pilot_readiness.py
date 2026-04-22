from __future__ import annotations

import json
from pathlib import Path

from inspection_system.app import pilot_readiness


def _write_capture_manifest(session_dir: Path, records: list[dict]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest = session_dir / "captures.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_build_supervised_pilot_status_reports_ready(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "projects" / "widget_a"
    config_dir = project_root / "config"
    reference_dir = project_root / "reference"
    config_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    active_paths = {
        "config_file": config_dir / "camera_config.json",
        "log_dir": project_root / "logs",
        "reference_dir": reference_dir,
        "reference_mask": reference_dir / "golden_reference_mask.png",
        "reference_image": reference_dir / "golden_reference_image.png",
    }

    tuning_records = [
        {"dataset_split": "tuning", "bucket": "good", "result_mismatch": False}
        for _ in range(10)
    ] + [
        {"dataset_split": "tuning", "bucket": "reject", "result_mismatch": False}
        for _ in range(5)
    ] + [
        {"dataset_split": "tuning", "bucket": "invalid_capture", "result_mismatch": False}
        for _ in range(2)
    ]
    validation_records = [
        {"dataset_split": "validation", "bucket": "good", "result_mismatch": False}
        for _ in range(5)
    ] + [
        {"dataset_split": "validation", "bucket": "reject", "result_mismatch": False}
        for _ in range(3)
    ] + [
        {"dataset_split": "validation", "bucket": "invalid_capture", "result_mismatch": False}
        for _ in range(2)
    ]
    regression_records = [
        {"dataset_split": "regression", "bucket": "good", "result_mismatch": False}
        for _ in range(5)
    ] + [
        {"dataset_split": "regression", "bucket": "reject", "result_mismatch": False}
        for _ in range(3)
    ] + [
        {"dataset_split": "regression", "bucket": "invalid_capture", "result_mismatch": False}
        for _ in range(2)
    ]

    _write_capture_manifest(project_root / "test_data" / "widget_a" / "cam_a" / "s1", tuning_records)
    _write_capture_manifest(project_root / "test_data" / "widget_a" / "cam_a" / "s2", validation_records)
    _write_capture_manifest(project_root / "test_data" / "widget_a" / "cam_a" / "s3", regression_records)

    monkeypatch.setattr(pilot_readiness, "get_current_project", lambda: "widget_a")
    monkeypatch.setattr(
        pilot_readiness,
        "list_runtime_reference_candidates",
        lambda config, active_paths=None: [{"reference_id": "golden"}],
    )
    monkeypatch.setattr(
        pilot_readiness,
        "get_commissioning_status",
        lambda config, active_paths=None, anomaly_detector=None: {
            "ready": True,
            "summary_line": "Commissioning: READY",
            "pending_good_records": 0,
            "actions": [],
        },
    )
    monkeypatch.setattr(pilot_readiness, "load_anomaly_detector", lambda active_paths: None)

    status = pilot_readiness.build_supervised_pilot_status({}, active_paths=active_paths)

    assert status["ready"] is True
    assert status["dataset"]["session_count"] == 3
    assert status["dataset"]["mismatch_counts"]["validation"] == 0
    assert status["phases"][-1]["ready"] is True


def test_build_supervised_pilot_status_reports_open_items(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "projects" / "pad_print"
    config_dir = project_root / "config"
    reference_dir = project_root / "reference"
    config_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    active_paths = {
        "config_file": config_dir / "camera_config.json",
        "log_dir": project_root / "logs",
        "reference_dir": reference_dir,
        "reference_mask": reference_dir / "golden_reference_mask.png",
        "reference_image": reference_dir / "golden_reference_image.png",
    }

    _write_capture_manifest(
        project_root / "test_data" / "pad_print" / "cam_b" / "s1",
        [
            {"dataset_split": "validation", "bucket": "good", "result_mismatch": True},
            {"dataset_split": "validation", "bucket": "reject", "result_mismatch": False},
        ],
    )

    monkeypatch.setattr(pilot_readiness, "get_current_project", lambda: "pad_print")
    monkeypatch.setattr(
        pilot_readiness,
        "list_runtime_reference_candidates",
        lambda config, active_paths=None: [],
    )
    monkeypatch.setattr(
        pilot_readiness,
        "get_commissioning_status",
        lambda config, active_paths=None, anomaly_detector=None: {
            "ready": False,
            "summary_line": "Commissioning: NOT READY",
            "warning": "Commissioning incomplete.",
            "pending_good_records": 2,
            "actions": ["Capture one golden reference before relying on production results."],
        },
    )
    monkeypatch.setattr(pilot_readiness, "load_anomaly_detector", lambda active_paths: None)

    status = pilot_readiness.build_supervised_pilot_status({}, active_paths=active_paths)

    assert status["ready"] is False
    assert any("No runtime references" in issue for issue in status["issues"])
    assert any("validation" in issue.lower() for issue in status["issues"])
    report_lines = pilot_readiness.format_supervised_pilot_report(status)
    assert report_lines[0] == "Supervised pilot readiness: NOT READY"
    assert any(line.startswith("- Engineering present") for line in report_lines)


def test_build_supervised_pilot_status_uses_configured_policy(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "projects" / "small_batch"
    config_dir = project_root / "config"
    reference_dir = project_root / "reference"
    config_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    active_paths = {
        "config_file": config_dir / "camera_config.json",
        "log_dir": project_root / "logs",
        "reference_dir": reference_dir,
        "reference_mask": reference_dir / "golden_reference_mask.png",
        "reference_image": reference_dir / "golden_reference_image.png",
    }

    _write_capture_manifest(
        project_root / "test_data" / "small_batch" / "cam_a" / "s1",
        [
            {"dataset_split": "tuning", "bucket": "good", "result_mismatch": False},
            {"dataset_split": "validation", "bucket": "good", "result_mismatch": False},
            {"dataset_split": "regression", "bucket": "good", "result_mismatch": False},
        ],
    )

    monkeypatch.setattr(pilot_readiness, "get_current_project", lambda: "small_batch")
    monkeypatch.setattr(
        pilot_readiness,
        "list_runtime_reference_candidates",
        lambda config, active_paths=None: [{"reference_id": "golden"}],
    )
    monkeypatch.setattr(
        pilot_readiness,
        "get_commissioning_status",
        lambda config, active_paths=None, anomaly_detector=None: {
            "ready": True,
            "summary_line": "Commissioning: READY",
            "pending_good_records": 0,
            "actions": [],
        },
    )
    monkeypatch.setattr(pilot_readiness, "load_anomaly_detector", lambda active_paths: None)

    status = pilot_readiness.build_supervised_pilot_status(
        {
            "pilot_readiness": {
                "targets": {
                    "tuning": {"good": 1, "reject": 0, "invalid_capture": 0},
                    "validation": {"good": 1, "reject": 0, "invalid_capture": 0},
                    "regression": {"good": 1, "reject": 0, "invalid_capture": 0},
                },
                "manual_floor_gates": [
                    "Supervisor confirms custom pilot cadence.",
                ],
            }
        },
        active_paths=active_paths,
    )

    assert status["ready"] is True
    assert status["manual_floor_gates"] == ["Supervisor confirms custom pilot cadence."]

    report_lines = pilot_readiness.format_supervised_pilot_report(status)
    assert "tuning: good 1/1 | reject 0/0 | invalid_capture 0/0 | borderline 0" in report_lines
    assert any(line == "- Supervisor confirms custom pilot cadence." for line in report_lines)
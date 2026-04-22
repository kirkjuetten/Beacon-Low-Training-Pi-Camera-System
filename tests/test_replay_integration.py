from __future__ import annotations

from pathlib import Path

import pytest

from inspection_system.app.dataset_diagnostics import resolve_project_context, simulate_training_episode
from inspection_system.app.replay_inspection import inspect_file
from inspection_system.app.result_status import CONFIG_ERROR, FAIL, PASS
from tests.replay_integration_harness import build_replay_project_fixture


@pytest.mark.integration
def test_inspect_file_replays_real_good_and_reject_samples(tmp_path: Path) -> None:
    project_root, session_dir = build_replay_project_fixture(
        tmp_path,
        selected_images=(
            "0009_good.png",
            "0011_reject_tail-tip-shape.png",
        ),
    )

    config, active_paths, project_context = resolve_project_context(session_dir)

    good_result = inspect_file(config, session_dir / "images" / "0009_good.png", active_paths=active_paths)
    reject_result = inspect_file(
        config,
        session_dir / "images" / "0011_reject_tail-tip-shape.png",
        active_paths=active_paths,
    )

    assert project_context == project_root
    assert good_result["status"] == PASS
    assert reject_result["status"] == FAIL
    assert good_result["reference_id"]
    assert reject_result["inspection_failure_cause"]
    assert (good_result.get("inspection_program") or {}).get("active_lane_id")
    assert reject_result["lane_results"]


@pytest.mark.integration
def test_simulate_training_episode_runs_against_compact_real_snapshot(tmp_path: Path) -> None:
    _, session_dir = build_replay_project_fixture(
        tmp_path,
        selected_images=(
            "0009_good.png",
            "0011_reject_tail-tip-shape.png",
            "0010_good.png",
            "0008_reject_light-pipe-position.png",
        ),
        split_overrides={
            "0009_good.png": "tuning",
            "0011_reject_tail-tip-shape.png": "tuning",
            "0010_good.png": "validation",
            "0008_reject_light-pipe-position.png": "validation",
        },
    )

    config, active_paths, _ = resolve_project_context(session_dir)
    report = simulate_training_episode(
        session_dir,
        config=config,
        active_paths=active_paths,
        duplicate_count=1,
        update_every=1,
        shuffle_seed=7,
    )

    assert report["training"]["processed_count"] == 2
    assert report["partition"]["training_unique_count"] == 2
    assert report["partition"]["evaluation_unique_count"] == 2
    assert report["evaluation"]["evaluated_count"] == 2
    assert report["training"]["update_events"]
    assert report["training"]["learning_record_count"] >= 2
    assert report["analysis"]["recommendations"]
    assert all(result["actual_status"] != CONFIG_ERROR for result in report["evaluation"]["results"])
    assert {Path(result["image_path"]).name for result in report["evaluation"]["results"]} == {
        "0010_good.png",
        "0008_reject_light-pipe-position.png",
    }
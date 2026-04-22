#!/usr/bin/env python3
"""Diagnostic training episode runner for collected test-photo sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspection_system.app.camera_interface import get_active_runtime_paths, load_config
from inspection_system.app.diagnostics_analysis import (
    CLOSE_MARGIN_THRESHOLDS,
    GATE_SPECS,
    _active_lane_id,
    _build_episode_analysis,
    _diagnose_result,
    _is_close_pass_margin,
    _primary_lane_id,
)
from inspection_system.app.diagnostics_episode import THRESHOLD_KEYS, simulate_training_episode_impl
from inspection_system.app.diagnostics_sources import (
    EPISODE_STATUS,
    EVALUATION_BUCKETS,
    TRAINING_BUCKETS,
    build_active_paths_from_project_root,
    build_diagnostic_output_path,
    duplicate_training_records,
    expected_status_for_record,
    load_capture_records,
    partition_episode_records,
    resolve_capture_manifests,
    resolve_project_context,
    resolve_project_root_from_source,
)
from inspection_system.app.interactive_training import ThresholdTrainer
from inspection_system.app.reference_service import clear_anomaly_training_artifacts, clear_reference_variants
from inspection_system.app.replay_inspection import inspect_file
from inspection_system.app.result_status import CONFIG_ERROR, FAIL, INVALID_CAPTURE, PASS, REGISTRATION_FAILED
from inspection_system.app.runtime_controller import get_commissioning_status, load_anomaly_detector


def simulate_training_episode(
    source_path: Path,
    *,
    config: dict | None = None,
    active_paths: dict | None = None,
    duplicate_count: int = 1,
    update_every: int = 5,
    shuffle_seed: int | None = None,
    train_splits: tuple[str, ...] = ("tuning",),
    eval_splits: tuple[str, ...] = ("validation", "regression"),
) -> dict:
    runtime_paths = active_paths or get_active_runtime_paths()
    return simulate_training_episode_impl(
        source_path,
        active_paths=runtime_paths,
        duplicate_count=duplicate_count,
        update_every=update_every,
        shuffle_seed=shuffle_seed,
        train_splits=train_splits,
        eval_splits=eval_splits,
        inspect_file_fn=inspect_file,
        threshold_trainer_cls=ThresholdTrainer,
        clear_reference_variants_fn=clear_reference_variants,
        clear_anomaly_training_artifacts_fn=clear_anomaly_training_artifacts,
        load_anomaly_detector_fn=load_anomaly_detector,
        get_commissioning_status_fn=get_commissioning_status,
    )


def save_episode_report(report: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sandboxed diagnostic training episode from collected test-photo sessions.")
    parser.add_argument("source", type=Path, help="Session directory, captures.jsonl path, or test_data root")
    parser.add_argument("--duplicates", type=int, default=2, help="How many times to replay the training pool during the episode")
    parser.add_argument("--update-every", type=int, default=5, help="How many simulated training parts between Update actions")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle seed for duplicate replay ordering")
    parser.add_argument("--train-split", dest="train_splits", action="append", default=None, help="Dataset split(s) to use for training, default: tuning")
    parser.add_argument("--eval-split", dest="eval_splits", action="append", default=None, help="Dataset split(s) to use for evaluation, default: validation and regression")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the JSON diagnostics report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train_splits = tuple(args.train_splits or ["tuning"])
    eval_splits = tuple(args.eval_splits or ["validation", "regression"])
    resolved_config, resolved_active_paths, project_root = resolve_project_context(args.source)
    report = simulate_training_episode(
        args.source,
        config=resolved_config,
        active_paths=resolved_active_paths,
        duplicate_count=args.duplicates,
        update_every=args.update_every,
        shuffle_seed=args.seed,
        train_splits=train_splits,
        eval_splits=eval_splits,
    )
    output_path = build_diagnostic_output_path(args.source, args.output)
    save_episode_report(report, output_path)
    print(json.dumps({
        "report_path": str(output_path),
        "project_root": str(project_root) if project_root is not None else None,
        "training_processed": report["training"]["processed_count"],
        "false_reject_rate": report["evaluation"]["false_reject_rate"],
        "false_accept_rate": report["evaluation"]["false_accept_rate"],
        "invalid_capture_miss_rate": report["evaluation"]["invalid_capture_miss_rate"],
        "commissioning_ready": report["training"]["commissioning_status"].get("ready"),
        "top_recommendation": (report.get("analysis", {}).get("recommendations") or [{}])[0].get("title"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
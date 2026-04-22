#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from inspection_system.app.diagnostics_analysis import _build_episode_analysis, _diagnose_result
from inspection_system.app.diagnostics_sources import (
    duplicate_training_records,
    expected_status_for_record,
    load_capture_records,
    partition_episode_records,
)


THRESHOLD_KEYS = [
    "min_required_coverage",
    "max_outside_allowed_ratio",
    "min_section_coverage",
    "max_mean_edge_distance_px",
    "max_section_edge_distance_px",
    "max_section_width_delta_ratio",
    "max_section_center_offset_px",
    "min_ssim",
    "max_mse",
    "min_anomaly_score",
]


def _copy_runtime_inputs(active_paths: dict, sandbox_root: Path) -> dict:
    config_dir = sandbox_root / "config"
    reference_dir = sandbox_root / "reference"
    log_dir = sandbox_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    source_config = Path(active_paths["config_file"])
    sandbox_config = config_dir / source_config.name
    shutil.copy2(source_config, sandbox_config)

    source_reference_dir = Path(active_paths["reference_dir"])
    if source_reference_dir.exists():
        for child in source_reference_dir.iterdir():
            target = reference_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)

    return {
        "config_file": sandbox_config,
        "reference_dir": reference_dir,
        "log_dir": log_dir,
        "reference_mask": reference_dir / "golden_reference_mask.png",
        "reference_image": reference_dir / "golden_reference_image.png",
    }


def _load_sandbox_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _snapshot_thresholds(config: dict) -> dict:
    inspection_cfg = config.get("inspection", {})
    return {key: inspection_cfg.get(key) for key in THRESHOLD_KEYS}


def _apply_training_update(trainer, config_state: dict) -> dict:
    pending_before = len(trainer.get_pending_records())
    pending_anomaly_samples = trainer.get_pending_anomaly_sample_count()
    suggestions = trainer.suggest_thresholds()
    learned_ranges = trainer.extract_learned_ranges()
    learning_update = trainer.apply_learning_update(config_state, suggestions, learned_ranges) if (suggestions or learned_ranges) else {
        "threshold_updates": {},
        "learned_ranges_saved": False,
        "learned_ranges_changed": False,
    }
    committed = trainer.commit_pending_feedback()
    rebuild_result = None
    inspection_mode = str(config_state.get("inspection", {}).get("inspection_mode", "mask_only")).strip().lower()
    if pending_anomaly_samples or inspection_mode in {"mask_and_ml", "full"}:
        rebuild_result = trainer.rebuild_anomaly_model(config_state)
    return {
        "pending_before_update": pending_before,
        "committed_count": committed,
        "threshold_updates": learning_update.get("threshold_updates", {}),
        "learned_ranges_saved": learning_update.get("learned_ranges_saved", False),
        "learned_ranges_changed": learning_update.get("learned_ranges_changed", False),
        "anomaly_rebuild": rebuild_result,
    }


def _evaluate_records(records: list[dict], config_state: dict, active_paths: dict, *, inspect_file_fn) -> dict:
    actual_counts = Counter()
    expected_counts = Counter()
    confusion = defaultdict(Counter)
    borderline_counts = Counter()
    results = []

    false_reject_count = 0
    false_accept_count = 0
    invalid_capture_miss_count = 0

    for record in records:
        image_path = Path(record["image_path"])
        result = inspect_file_fn(config_state, image_path, active_paths=active_paths)
        diagnosis = _diagnose_result(record, result)
        actual_status = str(result.get("status", "UNKNOWN"))
        bucket = str(record.get("bucket", "")).strip().lower()
        expected_status = expected_status_for_record(record)

        actual_counts[actual_status] += 1
        if expected_status:
            expected_counts[expected_status] += 1
            confusion[expected_status][actual_status] += 1
            if expected_status == "PASS" and actual_status != "PASS":
                false_reject_count += 1
            elif expected_status == "FAIL" and actual_status == "PASS":
                false_accept_count += 1
            elif expected_status == "INVALID_CAPTURE" and actual_status != "INVALID_CAPTURE":
                invalid_capture_miss_count += 1
        elif bucket == "borderline":
            borderline_counts[actual_status] += 1

        results.append(
            {
                "image_path": str(image_path),
                "bucket": bucket,
                "dataset_split": record.get("dataset_split"),
                "defect_category": record.get("defect_category"),
                "note": record.get("note"),
                "expected_status": expected_status,
                "actual_status": actual_status,
                "diagnosis": diagnosis,
                "result": result,
            }
        )

    good_total = expected_counts["PASS"]
    reject_total = expected_counts["FAIL"]
    invalid_total = expected_counts["INVALID_CAPTURE"]
    return {
        "evaluated_count": len(records),
        "expected_counts": dict(expected_counts),
        "actual_counts": dict(actual_counts),
        "confusion_matrix": {expected: dict(actuals) for expected, actuals in confusion.items()},
        "borderline_outcomes": dict(borderline_counts),
        "false_reject_count": false_reject_count,
        "false_accept_count": false_accept_count,
        "invalid_capture_miss_count": invalid_capture_miss_count,
        "false_reject_rate": round(false_reject_count / good_total, 4) if good_total else None,
        "false_accept_rate": round(false_accept_count / reject_total, 4) if reject_total else None,
        "invalid_capture_miss_rate": round(invalid_capture_miss_count / invalid_total, 4) if invalid_total else None,
        "results": results,
    }


def simulate_training_episode_impl(
    source_path: Path,
    *,
    active_paths: dict,
    duplicate_count: int,
    update_every: int,
    shuffle_seed: int | None,
    train_splits: tuple[str, ...],
    eval_splits: tuple[str, ...],
    inspect_file_fn,
    threshold_trainer_cls,
    clear_reference_variants_fn,
    clear_anomaly_training_artifacts_fn,
    load_anomaly_detector_fn,
    get_commissioning_status_fn,
) -> dict:
    records = load_capture_records(source_path)
    if not records:
        raise ValueError(f"No capture records found under {source_path}")

    training_records, evaluation_records, partition_info = partition_episode_records(
        records,
        train_splits=train_splits,
        eval_splits=eval_splits,
    )
    duplicated_training_records = duplicate_training_records(training_records, duplicate_count, shuffle_seed)

    with tempfile.TemporaryDirectory(prefix="beacon_episode_") as sandbox_dir:
        sandbox_root = Path(sandbox_dir)
        sandbox_paths = _copy_runtime_inputs(active_paths, sandbox_root)
        clear_reference_variants_fn(sandbox_paths)
        clear_anomaly_training_artifacts_fn(sandbox_paths)

        trainer = threshold_trainer_cls(Path(sandbox_paths["config_file"]), active_paths=sandbox_paths)
        config_state = _load_sandbox_config(Path(sandbox_paths["config_file"]))
        initial_thresholds = _snapshot_thresholds(config_state)

        update_every = max(1, int(update_every))
        update_events = []
        training_replay_mismatches = 0

        for index, record in enumerate(duplicated_training_records, start=1):
            image_path = Path(record["image_path"])
            replay_result = inspect_file_fn(config_state, image_path, active_paths=sandbox_paths)
            if expected_status_for_record(record) and replay_result.get("status") != expected_status_for_record(record):
                training_replay_mismatches += 1

            bucket = str(record.get("bucket", "")).strip().lower()
            feedback = "approve" if bucket == "good" else "reject"
            trainer.record_feedback(
                replay_result,
                feedback,
                label_info={
                    "final_class": "good" if bucket == "good" else "reject",
                    "defect_category": record.get("defect_category"),
                    "classification_reason": record.get("note"),
                },
                image_path=image_path,
            )

            if index % update_every == 0:
                update_snapshot = _apply_training_update(trainer, config_state)
                update_snapshot["after_training_record"] = index
                update_events.append(update_snapshot)
                config_state = _load_sandbox_config(Path(sandbox_paths["config_file"]))

        if trainer.get_pending_records():
            update_snapshot = _apply_training_update(trainer, config_state)
            update_snapshot["after_training_record"] = len(duplicated_training_records)
            update_events.append(update_snapshot)
            config_state = _load_sandbox_config(Path(sandbox_paths["config_file"]))

        final_thresholds = _snapshot_thresholds(config_state)
        threshold_drift = {
            key: {"initial": initial_thresholds.get(key), "final": final_thresholds.get(key)}
            for key in THRESHOLD_KEYS
            if initial_thresholds.get(key) != final_thresholds.get(key)
        }
        anomaly_detector = load_anomaly_detector_fn(sandbox_paths)
        commissioning_status = get_commissioning_status_fn(config_state, sandbox_paths, anomaly_detector)
        training_report = {
            "processed_count": len(duplicated_training_records),
            "training_replay_mismatch_count": training_replay_mismatches,
            "update_events": update_events,
            "pending_after_run": trainer.get_pending_summary(),
            "learning_record_count": len(trainer.get_learning_records()),
            "threshold_drift": threshold_drift,
            "final_thresholds": final_thresholds,
            "commissioning_status": commissioning_status,
        }
        evaluation = _evaluate_records(evaluation_records, config_state, sandbox_paths, inspect_file_fn=inspect_file_fn)
        analysis = _build_episode_analysis(training_report, evaluation)

        return {
            "source_path": str(source_path),
            "episode_parameters": {
                "duplicate_count": int(duplicate_count),
                "update_every": update_every,
                "shuffle_seed": shuffle_seed,
                "train_splits": list(train_splits),
                "eval_splits": list(eval_splits),
            },
            "partition": partition_info,
            "training": training_report,
            "evaluation": evaluation,
            "analysis": analysis,
        }
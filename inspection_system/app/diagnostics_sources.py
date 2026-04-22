#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from inspection_system.app.camera_interface import get_active_runtime_paths, load_config
from inspection_system.app.result_status import FAIL, INVALID_CAPTURE, PASS


EPISODE_STATUS = {"good": PASS, "reject": FAIL, "invalid_capture": INVALID_CAPTURE}
TRAINING_BUCKETS = {"good", "reject"}
EVALUATION_BUCKETS = {"good", "reject", "borderline", "invalid_capture"}


def resolve_capture_manifests(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    direct = path / "captures.jsonl"
    if direct.exists():
        return [direct]
    return sorted(path.rglob("captures.jsonl"))


def load_capture_records(path: Path) -> list[dict]:
    manifests = resolve_capture_manifests(path)
    records: list[dict] = []
    for manifest_path in manifests:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["manifest_path"] = str(manifest_path)
                session_dir = manifest_path.parent
                record.setdefault("session_dir", str(session_dir))
                image_path = record.get("image_path")
                if image_path:
                    candidate_path = Path(str(image_path))
                    if not candidate_path.exists():
                        relative_image_path = record.get("relative_image_path")
                        if relative_image_path:
                            local_image_path = session_dir / str(relative_image_path)
                            if local_image_path.exists():
                                record["image_path"] = str(local_image_path)
                records.append(record)
    return records


def expected_status_for_record(record: dict) -> str | None:
    explicit = record.get("expected_inspection_status")
    if explicit:
        return str(explicit)
    return EPISODE_STATUS.get(str(record.get("bucket", "")).strip().lower())


def partition_episode_records(
    records: list[dict],
    *,
    train_splits: tuple[str, ...] = ("tuning",),
    eval_splits: tuple[str, ...] = ("validation", "regression"),
) -> tuple[list[dict], list[dict], dict]:
    train_candidates = [
        record for record in records
        if str(record.get("bucket", "")).strip().lower() in TRAINING_BUCKETS
        and str(record.get("dataset_split", "")).strip().lower() in train_splits
    ]
    if not train_candidates:
        train_candidates = [
            record for record in records if str(record.get("bucket", "")).strip().lower() in TRAINING_BUCKETS
        ]

    eval_candidates = [
        record for record in records
        if str(record.get("bucket", "")).strip().lower() in EVALUATION_BUCKETS
        and str(record.get("dataset_split", "")).strip().lower() in eval_splits
    ]
    reused_training_for_eval = False
    if not eval_candidates:
        eval_candidates = [
            record for record in records if str(record.get("bucket", "")).strip().lower() in EVALUATION_BUCKETS
        ]
        reused_training_for_eval = True

    partition_info = {
        "training_unique_count": len(train_candidates),
        "evaluation_unique_count": len(eval_candidates),
        "evaluation_reused_training_pool": reused_training_for_eval,
    }
    return train_candidates, eval_candidates, partition_info


def duplicate_training_records(records: list[dict], duplicate_count: int, shuffle_seed: int | None = None) -> list[dict]:
    duplicate_count = max(1, int(duplicate_count))
    expanded: list[dict] = []
    for cycle_index in range(duplicate_count):
        for record in records:
            expanded.append({**record, "simulation_cycle": cycle_index + 1})
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(expanded)
    return expanded


def build_diagnostic_output_path(source_path: Path, output_path: Path | None = None) -> Path:
    if output_path is not None:
        return output_path
    base_dir = source_path if source_path.is_dir() else source_path.parent
    diagnostics_dir = base_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return diagnostics_dir / f"episode_{time.strftime('%Y%m%d_%H%M%S')}.json"


def resolve_project_root_from_source(source_path: Path) -> Path | None:
    current = source_path if source_path.is_dir() else source_path.parent
    for candidate in [current, *current.parents]:
        config_file = candidate / "config" / "camera_config.json"
        reference_dir = candidate / "reference"
        if config_file.exists() and reference_dir.exists():
            return candidate
        if candidate.name == "test_data":
            project_root = candidate.parent
            config_file = project_root / "config" / "camera_config.json"
            reference_dir = project_root / "reference"
            if config_file.exists() and reference_dir.exists():
                return project_root
    return None


def build_active_paths_from_project_root(project_root: Path) -> dict:
    reference_dir = project_root / "reference"
    return {
        "config_file": project_root / "config" / "camera_config.json",
        "log_dir": project_root / "logs",
        "reference_dir": reference_dir,
        "reference_mask": reference_dir / "golden_reference_mask.png",
        "reference_image": reference_dir / "golden_reference_image.png",
    }


def resolve_project_context(source_path: Path) -> tuple[dict, dict, Path | None]:
    project_root = resolve_project_root_from_source(source_path)
    if project_root is None:
        active_paths = get_active_runtime_paths()
        return load_config(), active_paths, None

    active_paths = build_active_paths_from_project_root(project_root)
    with Path(active_paths["config_file"]).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config, active_paths, project_root
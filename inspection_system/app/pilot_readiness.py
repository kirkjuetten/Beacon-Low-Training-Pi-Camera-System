from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from inspection_system.app.camera_interface import get_active_runtime_paths, get_current_project
from inspection_system.app.reference_service import list_runtime_reference_candidates
from inspection_system.app.runtime_controller import get_commissioning_status, load_anomaly_detector


CAPTURE_DATASET_SPLITS = ("tuning", "validation", "regression")
CAPTURE_BUCKETS = ("good", "reject", "borderline", "invalid_capture")

SUPERVISED_PILOT_TARGETS = {
    "tuning": {
        "good": 10,
        "reject": 5,
        "invalid_capture": 2,
    },
    "validation": {
        "good": 5,
        "reject": 3,
        "invalid_capture": 2,
    },
    "regression": {
        "good": 5,
        "reject": 3,
        "invalid_capture": 2,
    },
}

MANUAL_FLOOR_GATES = [
    "Engineering present at line start with authority to stop the run.",
    "Controlled challenge kit staged at the station and segregated from production parts.",
    "First lot executed as a supervised learning run with challenge inserts at a defined cadence.",
]


def _project_root_from_active_paths(active_paths: dict) -> Path:
    config_file = Path(active_paths["config_file"])
    if config_file.parent.name == "config":
        return config_file.parent.parent
    return config_file.parent


def _test_data_root_from_active_paths(active_paths: dict) -> Path:
    return _project_root_from_active_paths(active_paths) / "test_data"


def _empty_split_counts() -> dict[str, dict[str, int]]:
    return {
        split: {bucket: 0 for bucket in CAPTURE_BUCKETS}
        for split in CAPTURE_DATASET_SPLITS
    }


def summarize_test_data(active_paths: dict) -> dict:
    root = _test_data_root_from_active_paths(active_paths)
    summary = {
        "root": root,
        "exists": root.exists(),
        "session_count": 0,
        "record_count": 0,
        "invalid_manifest_lines": 0,
        "split_counts": _empty_split_counts(),
        "mismatch_count": 0,
        "mismatch_counts": {split: 0 for split in CAPTURE_DATASET_SPLITS},
    }

    if not root.exists():
        return summary

    for manifest_path in sorted(root.glob("**/captures.jsonl")):
        summary["session_count"] += 1
        try:
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            summary["invalid_manifest_lines"] += 1
            continue

        for raw_line in lines:
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                summary["invalid_manifest_lines"] += 1
                continue

            split = str(record.get("dataset_split", "tuning")).strip().lower()
            if split not in CAPTURE_DATASET_SPLITS:
                continue

            bucket = str(record.get("bucket", "good")).strip().lower()
            if bucket not in CAPTURE_BUCKETS:
                bucket = "good"

            summary["record_count"] += 1
            summary["split_counts"][split][bucket] += 1

            if bool(record.get("result_mismatch")):
                summary["mismatch_count"] += 1
                summary["mismatch_counts"][split] += 1

    return summary


def _format_split_count_line(split: str, counts: dict[str, int], targets: dict[str, int]) -> str:
    parts = []
    for bucket in ("good", "reject", "invalid_capture", "borderline"):
        value = counts.get(bucket, 0)
        target = targets.get(bucket)
        if target is None:
            parts.append(f"{bucket} {value}")
        else:
            parts.append(f"{bucket} {value}/{target}")
    return f"{split}: " + " | ".join(parts)


def build_supervised_pilot_status(
    config: dict,
    active_paths: Optional[dict] = None,
    anomaly_detector=None,
) -> dict:
    resolved_active_paths = active_paths or get_active_runtime_paths()
    resolved_anomaly_detector = anomaly_detector
    if resolved_anomaly_detector is None:
        resolved_anomaly_detector = load_anomaly_detector(resolved_active_paths)

    current_project = get_current_project()
    commissioning = get_commissioning_status(config, resolved_active_paths, resolved_anomaly_detector)
    reference_candidates = list_runtime_reference_candidates(config, resolved_active_paths)
    dataset_summary = summarize_test_data(resolved_active_paths)

    issues: list[str] = []
    actions: list[str] = []
    warnings: list[str] = []

    recipe_isolated = bool(current_project)
    if not recipe_isolated:
        issues.append("No active project is selected. Pilot work must run in a project-scoped recipe.")
        actions.append("Create or switch to a dedicated project before commissioning the floor pilot.")

    reference_ready = bool(reference_candidates)
    if not reference_ready:
        issues.append("No runtime references are available for the active project.")
        actions.append("Capture the golden reference and any approved-good variants required by the recipe.")

    commissioning_ready = bool(commissioning.get("ready", False)) and reference_ready
    if not commissioning_ready:
        if commissioning.get("warning"):
            issues.append(str(commissioning["warning"]))
        for action in commissioning.get("actions", []):
            if action not in actions:
                actions.append(action)

    dataset_targets_met = True
    for split, targets in SUPERVISED_PILOT_TARGETS.items():
        counts = dataset_summary["split_counts"][split]
        for bucket, target in targets.items():
            if counts.get(bucket, 0) < target:
                dataset_targets_met = False
                issues.append(
                    f"Controlled challenge set incomplete for {split}: need {target} {bucket} captures, have {counts.get(bucket, 0)}."
                )
                actions.append(
                    f"Collect more {split} {bucket} captures until the supervised pilot target is met."
                )

    validation_clean = dataset_summary["mismatch_counts"]["validation"] == 0
    regression_clean = dataset_summary["mismatch_counts"]["regression"] == 0
    if not validation_clean:
        issues.append(
            f"Validation replay still has {dataset_summary['mismatch_counts']['validation']} expected-vs-actual mismatches."
        )
        actions.append("Review validation mismatches and retune the recipe before floor launch.")
    if not regression_clean:
        issues.append(
            f"Regression replay still has {dataset_summary['mismatch_counts']['regression']} expected-vs-actual mismatches."
        )
        actions.append("Review regression mismatches and retune the recipe before floor launch.")

    pending_good_records = int(commissioning.get("pending_good_records", 0))
    learning_run_closed = pending_good_records == 0
    if not learning_run_closed:
        issues.append(f"Learning run is still open with {pending_good_records} pending approved-good records.")
        actions.append("Press Update in training to commit the pending learning run before floor launch.")

    tuning_mismatches = dataset_summary["mismatch_counts"]["tuning"]
    if tuning_mismatches:
        warnings.append(
            f"Tuning split still contains {tuning_mismatches} mismatches. This is acceptable during learning, but document the final decision before launch."
        )

    technical_ready = all(
        [
            recipe_isolated,
            reference_ready,
            commissioning_ready,
            dataset_targets_met,
            validation_clean,
            regression_clean,
            learning_run_closed,
        ]
    )

    phases = [
        {
            "name": "Recipe Isolation",
            "ready": recipe_isolated,
            "summary": (
                f"Project '{current_project}' is active."
                if recipe_isolated
                else "Project-scoped recipe has not been selected yet."
            ),
        },
        {
            "name": "Commission Baseline",
            "ready": commissioning_ready,
            "summary": str(commissioning.get("summary_line", "Commissioning status unavailable.")),
        },
        {
            "name": "Controlled Challenge Set",
            "ready": dataset_targets_met and validation_clean and regression_clean,
            "summary": (
                f"{dataset_summary['session_count']} capture sessions, {dataset_summary['record_count']} labeled images, "
                f"validation mismatches {dataset_summary['mismatch_counts']['validation']}, regression mismatches {dataset_summary['mismatch_counts']['regression']}."
            ),
        },
        {
            "name": "Learning Run Closed",
            "ready": learning_run_closed,
            "summary": (
                "Training update is committed and the supervised learning run is closed."
                if learning_run_closed
                else f"Pending approved-good records remain: {pending_good_records}."
            ),
        },
        {
            "name": "Supervised Pilot Launch",
            "ready": technical_ready,
            "summary": (
                "Technical gate is closed and the recipe is ready for a supervised floor pilot."
                if technical_ready
                else "Technical gate is still open. Finish the actions below before the first floor pilot."
            ),
        },
    ]

    return {
        "ready": technical_ready,
        "current_project": current_project,
        "active_paths": resolved_active_paths,
        "reference_candidate_count": len(reference_candidates),
        "commissioning": commissioning,
        "dataset": dataset_summary,
        "issues": issues,
        "actions": list(dict.fromkeys(actions)),
        "warnings": warnings,
        "phases": phases,
        "manual_floor_gates": list(MANUAL_FLOOR_GATES),
    }


def format_supervised_pilot_report(status: dict) -> list[str]:
    readiness = "READY" if status.get("ready") else "NOT READY"
    current_project = status.get("current_project") or "<none>"
    dataset = status.get("dataset", {})
    commissioning = status.get("commissioning", {})
    lines = [
        f"Supervised pilot readiness: {readiness}",
        f"Project: {current_project}",
        f"Reference candidates: {status.get('reference_candidate_count', 0)}",
        str(commissioning.get("summary_line", "Commissioning status unavailable.")),
        f"Test data root: {dataset.get('root')}",
        f"Captured sessions: {dataset.get('session_count', 0)} | labeled images: {dataset.get('record_count', 0)}",
    ]

    split_counts = dataset.get("split_counts", {})
    for split in CAPTURE_DATASET_SPLITS:
        lines.append(
            _format_split_count_line(
                split,
                split_counts.get(split, {}),
                SUPERVISED_PILOT_TARGETS.get(split, {}),
            )
        )

    lines.append("Phases:")
    for index, phase in enumerate(status.get("phases", []), start=1):
        state = "READY" if phase.get("ready") else "OPEN"
        lines.append(f"{index}. {phase.get('name')}: {state} - {phase.get('summary')}")

    warnings = status.get("warnings", [])
    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")

    issues = status.get("issues", [])
    if issues:
        lines.append("Blocking issues:")
        for issue in issues:
            lines.append(f"- {issue}")

    actions = status.get("actions", [])
    if actions:
        lines.append("Next actions:")
        for action in actions:
            lines.append(f"- {action}")

    lines.append("Manual floor gates:")
    for gate in status.get("manual_floor_gates", []):
        lines.append(f"- {gate}")

    return lines


def print_supervised_pilot_report(config: dict, active_paths: Optional[dict] = None) -> int:
    status = build_supervised_pilot_status(config, active_paths=active_paths)
    for line in format_supervised_pilot_report(status):
        print(line)
    return 0 if status.get("ready") else 1
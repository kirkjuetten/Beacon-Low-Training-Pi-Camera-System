#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional

from inspection_system.app.anomaly_detection_utils import AnomalyDetector
from inspection_system.app.alignment_utils import align_sample_mask
from inspection_system.app.frame_acquisition import capture_to_temp, cleanup_temp_image
from inspection_system.app.inspection_pipeline import inspect_against_references
from inspection_system.app.morphology_utils import dilate_mask, erode_mask
from inspection_system.app.preprocessing_utils import make_binary_mask
from inspection_system.app.reference_service import (
    MIN_ANOMALY_TRAINING_SAMPLES,
    build_registration_commissioning_summary,
    get_anomaly_model_artifact_paths,
    get_anomaly_model_metadata,
    get_reference_variant_directories,
    list_anomaly_training_samples,
    load_reference_metadata,
    registration_baseline_matches_config,
)
from inspection_system.app.reference_region_utils import build_reference_regions
from inspection_system.app.reference_service import list_runtime_reference_candidates, save_debug_outputs
from inspection_system.app.scoring_utils import evaluate_metrics, normalize_inspection_mode, score_sample
from inspection_system.app.section_mask_utils import compute_section_masks
from inspection_system.app.inspection_runtime_context import build_inspection_runtime_context
from inspection_system.app.training_labels import resolve_learning_class
from inspection_system.app.camera_interface import import_cv2_and_numpy, get_active_runtime_paths


COMMISSIONING_DEFAULTS = {
    "golden_only": {
        "min_good_samples": 10,
        "min_active_variants": 0,
        "requires_golden_reference": True,
    },
    "hybrid": {
        "min_good_samples": 8,
        "min_active_variants": 3,
        "requires_golden_reference": True,
    },
    "multi_good_experimental": {
        "min_good_samples": 6,
        "min_active_variants": 6,
        "requires_golden_reference": False,
    },
}

COMMISSIONING_WORKFLOW_TOTAL_STAGES = 5


def _optional_float(value):
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value, default: int) -> int:
    if value in {None, ""}:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _append_unique(lines: list[str], text: str) -> None:
    if text and text not in lines:
        lines.append(text)


def _load_training_records(active_paths: dict) -> list[dict]:
    config_path = active_paths.get("config_file")
    if config_path is None:
        return []

    training_file = Path(config_path).parent / "training_data.json"
    if not training_file.exists():
        return []

    try:
        payload = json.loads(training_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _count_good_training_records(active_paths: dict) -> dict[str, int]:
    counts = {
        "committed_good_records": 0,
        "pending_good_records": 0,
    }
    for record in _load_training_records(active_paths):
        if resolve_learning_class(record) != "good":
            continue
        learning_state = str(record.get("learning_state", "committed")).strip().lower()
        if learning_state == "pending":
            counts["pending_good_records"] += 1
        elif learning_state != "discarded":
            counts["committed_good_records"] += 1
    return counts


def _count_reference_variants(active_paths: dict, state: str) -> int:
    variant_dirs = get_reference_variant_directories(active_paths)
    base_dir = variant_dirs.get(state)
    if base_dir is None or not base_dir.exists():
        return 0
    return sum(1 for path in base_dir.iterdir() if path.is_dir())


def _get_commissioning_requirements(config: dict) -> dict:
    inspection_cfg = config.get("inspection", {})
    training_cfg = config.get("training", {})
    reference_strategy = str(inspection_cfg.get("reference_strategy", "golden_only")).strip().lower() or "golden_only"
    defaults = COMMISSIONING_DEFAULTS.get(reference_strategy, COMMISSIONING_DEFAULTS["golden_only"])
    return {
        "reference_strategy": reference_strategy,
        "min_good_samples": _optional_int(
            training_cfg.get(f"{reference_strategy}_min_good_samples"),
            defaults["min_good_samples"],
        ),
        "min_active_variants": _optional_int(
            training_cfg.get(f"{reference_strategy}_min_active_variants"),
            defaults["min_active_variants"],
        ),
        "requires_golden_reference": bool(
            training_cfg.get(
                f"{reference_strategy}_requires_golden_reference",
                defaults["requires_golden_reference"],
            )
        ),
    }


def _has_commissioning_paths(active_paths: Optional[dict]) -> bool:
    return bool(active_paths) and all(
        key in active_paths for key in ("config_file", "reference_dir", "reference_mask", "reference_image")
    )


def _annotate_commissioning_workflow(status: dict) -> dict:
    stage_index = 1
    stage_title = "Setup"
    instruction = "Entering training workflow. Capture or confirm the project setup before collecting parts."
    upgrade_prompt = None

    min_good_samples = int(status.get("min_good_samples", 0))
    committed_good_records = int(status.get("committed_good_records", 0))
    pending_good_records = int(status.get("pending_good_records", 0))
    min_active_variants = int(status.get("min_active_variants", 0))
    active_reference_variants = int(status.get("active_reference_variants", 0))
    pending_reference_variants = int(status.get("pending_reference_variants", 0))
    reference_strategy = str(status.get("reference_strategy", "golden_only"))

    if status.get("requires_golden_reference") and not status.get("golden_reference_present"):
        stage_title = "Capture Golden Reference"
        instruction = "Capture one golden reference. Then load a clearly known-good part to verify the setup."
    elif not status.get("registration_ready", True):
        stage_index = 2
        stage_title = "Registration Setup"
        instruction = (
            status.get("registration_actions", []) or [
                "Complete the registration anchor and datum setup before relying on production results."
            ]
        )[0]
    elif status.get("golden_reference_present") and not status.get("registration_baseline_captured", False):
        stage_index = 2
        stage_title = "Capture Registration Baseline"
        instruction = (
            status.get("actions", []) or [
                "Re-capture the golden reference to stamp the current registration baseline."
            ]
        )[0]
    elif committed_good_records <= 0:
        stage_index = 2
        stage_title = "Golden Check"
        instruction = "Load known-good part and approve only clearly good samples to verify the reference."
    elif committed_good_records < min_good_samples:
        stage_index = 3
        if pending_good_records and committed_good_records + pending_good_records >= min_good_samples:
            stage_title = "Commit Baseline"
            instruction = (
                f"Baseline captured. Press Update to commit {pending_good_records} pending approved-good parts."
            )
        else:
            remaining = max(0, min_good_samples - committed_good_records)
            stage_title = "Baseline Build"
            instruction = (
                f"Load known-good part. More good examples needed: {remaining} of {min_good_samples} remaining."
            )
    elif min_active_variants > 0 and active_reference_variants < min_active_variants:
        stage_index = 4
        if pending_reference_variants and active_reference_variants + pending_reference_variants >= min_active_variants:
            stage_title = "Activate Variation Library"
            instruction = (
                f"Variation library is staged. Press Update to activate {pending_reference_variants} pending approved-good references."
            )
        else:
            remaining = max(0, min_active_variants - active_reference_variants)
            stage_title = "Variation Library"
            instruction = (
                f"Load known-good molded part. More approved-good references needed: {remaining} of {min_active_variants} remaining."
            )
    elif not status.get("ml_ready", True):
        stage_index = 4
        stage_title = "Refresh ML Model"
        instruction = "Press Update to rebuild the anomaly model before relying on ML-backed production gating."
    elif not status.get("ready", False):
        stage_index = 4
        stage_title = "Review Pending Updates"
        instruction = status.get("actions", ["Review the pending commissioning actions before proceeding."])[0]
    else:
        stage_index = COMMISSIONING_WORKFLOW_TOTAL_STAGES
        stage_title = "Production Ready"
        instruction = "Commissioning complete. Inspect parts in production mode or continue collecting approved-good examples."
        hybrid_min_good = COMMISSIONING_DEFAULTS["hybrid"]["min_good_samples"]
        if reference_strategy == "golden_only" and committed_good_records >= hybrid_min_good:
            upgrade_prompt = "Hybrid now available. Activate if molded-part variation needs multiple approved-good references."

    status.update(
        {
            "workflow_stage_index": stage_index,
            "workflow_stage_total": COMMISSIONING_WORKFLOW_TOTAL_STAGES,
            "workflow_stage_title": stage_title,
            "workflow_instruction": instruction,
            "workflow_upgrade_prompt": upgrade_prompt,
        }
    )
    return status


def get_commissioning_status(
    config: dict,
    active_paths: Optional[dict] = None,
    anomaly_detector: Optional[AnomalyDetector] = None,
) -> dict:
    requirements = _get_commissioning_requirements(config)
    status = {
        **requirements,
        "golden_reference_present": False,
        "runtime_reference_count": 0,
        "active_reference_variants": 0,
        "pending_reference_variants": 0,
        "committed_good_records": 0,
        "pending_good_records": 0,
        "ml_ready": True,
        "registration_ready": True,
        "registration_baseline_captured": False,
        "registration_summary": build_registration_commissioning_summary(config).get("summary", "registration unknown"),
        "registration_issues": [],
        "registration_actions": [],
        "summary_line": "Commissioning: training targets unavailable",
        "issues": [],
        "actions": [],
        "warning": None,
        "ready": False,
        "workflow_stage_index": 1,
        "workflow_stage_total": COMMISSIONING_WORKFLOW_TOTAL_STAGES,
        "workflow_stage_title": "Setup",
        "workflow_instruction": "Entering training workflow. Capture or confirm the project setup before collecting parts.",
        "workflow_upgrade_prompt": None,
    }

    if not _has_commissioning_paths(active_paths):
        status["summary_line"] = (
            "Commissioning: "
            f"target {status['min_good_samples']} approved-good parts"
        )
        return _annotate_commissioning_workflow(status)

    runtime_refs = list_runtime_reference_candidates(config, active_paths)
    training_counts = _count_good_training_records(active_paths)
    anomaly_status = get_anomaly_model_status(config, anomaly_detector, active_paths)
    golden_reference_present = Path(active_paths["reference_mask"]).exists() and Path(active_paths["reference_image"]).exists()
    active_reference_variants = _count_reference_variants(active_paths, "active")
    pending_reference_variants = _count_reference_variants(active_paths, "pending")

    issues: list[str] = []
    actions: list[str] = []
    follow_up_actions: list[str] = []

    registration_summary = build_registration_commissioning_summary(config)
    registration_baseline_captured = False
    if golden_reference_present:
        reference_metadata = load_reference_metadata(Path(active_paths["reference_dir"]) / "ref_meta.json") or {}
        baseline_from_metadata = reference_metadata.get("registration_baseline")
        if registration_baseline_matches_config(baseline_from_metadata, registration_summary):
            registration_baseline_captured = True
        else:
            _append_unique(follow_up_actions, "Re-capture the golden reference to stamp the current registration baseline.")

        if not bool(registration_summary.get("ready", True)):
            issues.append("registration setup incomplete")
            for action in registration_summary.get("actions", []):
                _append_unique(actions, action)
        elif not registration_baseline_captured:
            issues.append("registration baseline not captured")

    if status["requires_golden_reference"] and not golden_reference_present:
        issues.append("golden reference missing")
        _append_unique(actions, "Capture one golden reference before relying on production results.")

    committed_good_records = training_counts["committed_good_records"]
    pending_good_records = training_counts["pending_good_records"]
    min_good_samples = status["min_good_samples"]
    if committed_good_records < min_good_samples:
        issues.append(f"approved-good baseline {committed_good_records}/{min_good_samples}")
        if pending_good_records:
            _append_unique(actions, f"Press Update to commit {pending_good_records} pending approved-good parts.")
        remaining_good = max(0, min_good_samples - committed_good_records - pending_good_records)
        if remaining_good:
            _append_unique(actions, f"Collect {remaining_good} more approved-good parts.")
    elif pending_good_records:
        _append_unique(actions, f"Press Update to commit {pending_good_records} pending approved-good parts.")

    min_active_variants = status["min_active_variants"]
    if min_active_variants > 0 and active_reference_variants < min_active_variants:
        issues.append(f"active approved-good references {active_reference_variants}/{min_active_variants}")
        if pending_reference_variants:
            _append_unique(actions, f"Press Update to activate {pending_reference_variants} pending approved-good references.")
        remaining_variants = max(0, min_active_variants - active_reference_variants - pending_reference_variants)
        if remaining_variants:
            _append_unique(actions, f"Capture {remaining_variants} more approved-good reference variants.")
    elif pending_reference_variants:
        _append_unique(actions, f"Press Update to activate {pending_reference_variants} pending approved-good references.")

    ml_ready = True
    if anomaly_status["ml_mode_selected"] and not anomaly_status["ready"]:
        ml_ready = False
        if anomaly_status["pending_good_samples"] or anomaly_status["model_stale"]:
            _append_unique(actions, "Press Update to rebuild the anomaly model for the current approved-good sample library.")

    for action in follow_up_actions:
        _append_unique(actions, action)

    ready = not issues and ml_ready
    summary_parts = []
    if status["requires_golden_reference"]:
        summary_parts.append("golden ok" if golden_reference_present else "golden missing")
    summary_parts.append("reg ok" if registration_summary.get("ready", True) else "reg setup")
    if golden_reference_present:
        summary_parts.append("baseline ok" if registration_baseline_captured else "baseline pending")
    summary_parts.append(f"good {committed_good_records}/{min_good_samples}")
    if min_active_variants > 0:
        summary_parts.append(f"refs {active_reference_variants}/{min_active_variants}")
    if anomaly_status["ml_mode_selected"]:
        summary_parts.append("ml ready" if ml_ready else "ml pending")

    status.update(
        {
            "golden_reference_present": golden_reference_present,
            "runtime_reference_count": len(runtime_refs),
            "active_reference_variants": active_reference_variants,
            "pending_reference_variants": pending_reference_variants,
            "committed_good_records": committed_good_records,
            "pending_good_records": pending_good_records,
            "ml_ready": ml_ready,
            "registration_ready": bool(registration_summary.get("ready", True)),
            "registration_baseline_captured": registration_baseline_captured,
            "registration_summary": str(registration_summary.get("summary", "registration unknown")),
            "registration_issues": list(registration_summary.get("issues", [])),
            "registration_actions": list(registration_summary.get("actions", [])),
            "issues": issues,
            "actions": actions,
            "ready": ready,
            "summary_line": (
                f"Commissioning: {'READY' if ready else 'NOT READY'}"
                + (f" | {' | '.join(summary_parts)}" if summary_parts else "")
            ),
        }
    )

    if issues:
        action_text = f" {actions[0]}" if actions else ""
        status["warning"] = (
            f"Commissioning is incomplete for {status['reference_strategy']}: "
            f"{'; '.join(issues)}.{action_text}"
        )
    return _annotate_commissioning_workflow(status)


def format_commissioning_status_lines(status: dict) -> list[str]:
    lines = [
        "Workflow: "
        f"Stage {status.get('workflow_stage_index', 1)}/{status.get('workflow_stage_total', COMMISSIONING_WORKFLOW_TOTAL_STAGES)}"
        f" - {status.get('workflow_stage_title', 'Setup')}",
        f"Instruction: {status.get('workflow_instruction', 'Review the commissioning steps for this project.')}",
        f"Registration: {status.get('registration_summary', 'registration unknown')}",
        status.get("summary_line", "Commissioning: unknown"),
    ]
    upgrade_prompt = status.get("workflow_upgrade_prompt")
    if upgrade_prompt:
        lines.append(f"Prompt: {upgrade_prompt}")
    for action in status.get("actions", [])[:2]:
        lines.append(f"Next: {action}")
    return lines


def describe_edge_gate_status(config: dict) -> tuple[str, str | None]:
    inspection_cfg = config.get("inspection", {})
    max_mean_edge_distance_px = _optional_float(inspection_cfg.get("max_mean_edge_distance_px"))
    max_section_edge_distance_px = _optional_float(inspection_cfg.get("max_section_edge_distance_px"))

    mean_status = (
        f"global<={max_mean_edge_distance_px:.2f}px" if max_mean_edge_distance_px is not None else "global off"
    )
    section_status = (
        f"section<={max_section_edge_distance_px:.2f}px"
        if max_section_edge_distance_px is not None
        else "section off"
    )
    status_line = f"Edge Gates: {mean_status} | {section_status}"

    if max_mean_edge_distance_px is None and max_section_edge_distance_px is None:
        return (
            status_line,
            "Hint: set Max Mean Edge Distance and Max Section Edge Distance to enable global and section edge drift checks.",
        )
    if max_mean_edge_distance_px is None:
        return (
            status_line,
            "Hint: set Max Mean Edge Distance to enable the global edge drift gate.",
        )
    if max_section_edge_distance_px is None:
        return (
            status_line,
            "Hint: set Max Section Edge Distance to enable the section edge drift gate.",
        )
    return status_line, None


def describe_section_width_gate_status(config: dict) -> tuple[str, str | None]:
    inspection_cfg = config.get("inspection", {})
    max_section_width_delta_ratio = _optional_float(inspection_cfg.get("max_section_width_delta_ratio"))
    if max_section_width_delta_ratio is None:
        return (
            "Width Gate: section off",
            "Hint: set Max Section Width Drift to enable per-section width drift checks.",
        )
    return (f"Width Gate: section<={max_section_width_delta_ratio:.1%}", None)


def describe_section_center_gate_status(config: dict) -> tuple[str, str | None]:
    inspection_cfg = config.get("inspection", {})
    max_section_center_offset_px = _optional_float(inspection_cfg.get("max_section_center_offset_px"))
    if max_section_center_offset_px is None:
        return (
            "Center Gate: section off",
            "Hint: set Max Section Center Offset to enable per-section center drift checks.",
        )
    return (f"Center Gate: section<={max_section_center_offset_px:.2f}px", None)


def load_anomaly_detector(active_paths: dict):
    model_path = Path(active_paths["reference_dir"]) / "anomaly_model.pkl"
    model_path = get_anomaly_model_artifact_paths(active_paths)["model"]
    if not model_path.exists():
        return None

    detector = AnomalyDetector(model_path=model_path)
    try:
        detector.load_model()
        return detector
    except Exception as exc:
        print(f"Warning: failed to load anomaly model from {model_path}: {exc}")
        return None


def get_anomaly_model_status(
    config: dict,
    anomaly_detector: Optional[AnomalyDetector],
    active_paths: Optional[dict] = None,
) -> dict:
    inspection_cfg = config.get("inspection", {})
    inspection_mode = normalize_inspection_mode(inspection_cfg.get("inspection_mode", "mask_only"))
    status = {
        "inspection_mode": inspection_mode,
        "ml_mode_selected": inspection_mode in {"mask_and_ml", "full"},
        "active_good_samples": 0,
        "pending_good_samples": 0,
        "minimum_required_samples": MIN_ANOMALY_TRAINING_SAMPLES,
        "model_path": None,
        "model_exists": False,
        "trained_sample_count": None,
        "model_stale": False,
        "ready": False,
    }

    if active_paths is None:
        status["model_exists"] = anomaly_detector is not None
        status["ready"] = anomaly_detector is not None
        return status

    artifact_paths = get_anomaly_model_artifact_paths(active_paths)
    metadata = get_anomaly_model_metadata(active_paths) or {}
    active_samples = list_anomaly_training_samples(active_paths, states=("active",))
    pending_samples = list_anomaly_training_samples(active_paths, states=("pending",))

    trained_sample_count = metadata.get("trained_sample_count")
    if trained_sample_count is not None:
        trained_sample_count = int(trained_sample_count)

    model_exists = artifact_paths["model"].exists()
    model_stale = False
    if model_exists and trained_sample_count is not None:
        model_stale = trained_sample_count != len(active_samples)

    status.update(
        {
            "active_good_samples": len(active_samples),
            "pending_good_samples": len(pending_samples),
            "model_path": str(artifact_paths["model"]),
            "model_exists": model_exists,
            "trained_sample_count": trained_sample_count,
            "model_stale": model_stale,
            "ready": (
                model_exists
                and anomaly_detector is not None
                and len(active_samples) >= MIN_ANOMALY_TRAINING_SAMPLES
                and not model_stale
            ),
        }
    )
    return status


def get_inspection_runtime_warnings(
    config: dict,
    anomaly_detector: Optional[AnomalyDetector],
    active_paths: Optional[dict] = None,
) -> list[str]:
    inspection_cfg = config.get("inspection", {})
    inspection_mode = normalize_inspection_mode(inspection_cfg.get("inspection_mode", "mask_only"))
    warnings: list[str] = []
    anomaly_status = get_anomaly_model_status(config, anomaly_detector, active_paths)

    if inspection_mode in {"mask_and_ml", "full"}:
        if active_paths is not None and anomaly_status["active_good_samples"] < anomaly_status["minimum_required_samples"]:
            warnings.append(
                "ML-backed mode is selected but there are not enough approved-good samples to train the anomaly model. "
                f"Committed good samples: {anomaly_status['active_good_samples']}/{anomaly_status['minimum_required_samples']}."
            )
        elif active_paths is not None and anomaly_status["model_stale"]:
            warnings.append(
                "ML-backed mode is selected but the anomaly model is stale for the current approved-good sample library. "
                "Press Update in training to rebuild it."
            )
        elif anomaly_detector is None:
            warnings.append(
                "ML-backed mode is selected but no trained anomaly model is available. The anomaly check will not be enforced."
            )
        if inspection_cfg.get("min_anomaly_score") in {None, ""}:
            warnings.append(
                "ML-backed mode is selected but Min Anomaly Score is not set. The anomaly gate is inactive."
            )

    commissioning_status = get_commissioning_status(config, active_paths, anomaly_detector)
    if commissioning_status.get("warning"):
        warnings.append(commissioning_status["warning"])

    return warnings


def format_operator_mode_lines(
    config: dict,
    active_paths: Optional[dict] = None,
    anomaly_detector: Optional[AnomalyDetector] = None,
) -> list[str]:
    inspection_cfg = config.get("inspection", {})
    inspection_mode = normalize_inspection_mode(inspection_cfg.get("inspection_mode", "mask_only"))
    reference_strategy = str(inspection_cfg.get("reference_strategy", "golden_only")).strip().lower() or "golden_only"
    blend_mode = str(inspection_cfg.get("blend_mode", "hard_only")).strip().lower() or "hard_only"
    tolerance_mode = str(inspection_cfg.get("tolerance_mode", "balanced")).strip().lower() or "balanced"
    lines = [
        f"Mode: {inspection_mode} | Ref: {reference_strategy}",
        f"Blend: {blend_mode} | Tol: {tolerance_mode}",
    ]
    edge_status_line, edge_hint = describe_edge_gate_status(config)
    lines.append(edge_status_line)
    if edge_hint:
        lines.append(edge_hint)
    width_status_line, width_hint = describe_section_width_gate_status(config)
    lines.append(width_status_line)
    if width_hint:
        lines.append(width_hint)
    center_status_line, center_hint = describe_section_center_gate_status(config)
    lines.append(center_status_line)
    if center_hint:
        lines.append(center_hint)

    commissioning_status = get_commissioning_status(config, active_paths, anomaly_detector)
    lines.extend(format_commissioning_status_lines(commissioning_status))

    if active_paths is not None:
        reference_count = len(list_runtime_reference_candidates(config, active_paths))
        lines[0] = f"Mode: {inspection_mode} | Ref: {reference_strategy} ({reference_count})"
        anomaly_status = get_anomaly_model_status(config, anomaly_detector, active_paths)
        if anomaly_status["ml_mode_selected"]:
            if anomaly_status["ready"]:
                lines.append(f"ML: ready ({anomaly_status['active_good_samples']} approved-good samples)")
            else:
                lines.append(
                    "ML: "
                    f"{anomaly_status['active_good_samples']}/{anomaly_status['minimum_required_samples']} approved-good samples"
                )

    return lines


def print_inspection_runtime_warnings(config: dict, anomaly_detector: Optional[AnomalyDetector]) -> list[str]:
    warnings = get_inspection_runtime_warnings(config, anomaly_detector)
    for warning in warnings:
        print(f"Warning: {warning}")
    return warnings


def run_interactive_training(config: dict) -> int:
    """Import and run interactive training mode."""
    try:
        from inspection_system.app.interactive_training import run_interactive_training as train_func
        return train_func(config)
    except ImportError as exc:
        print(f"Interactive training not available: {exc}")
        return 1


def run_production_mode(config: dict, indicator) -> int:
    """Import and run production inspection mode."""
    try:
        from inspection_system.app.production_screen import run_production_mode as production_func

        return production_func(config, indicator)
    except ImportError as exc:
        print(f"Production mode not available: {exc}")
        return 1


def print_inspection_result(passed: bool, details: dict) -> None:
    print("Inspection result:", "PASS" if passed else "FAIL")
    print(f"Inspection mode: {details.get('inspection_mode', 'mask_only')}")
    registration = details.get("registration", {}) if isinstance(details.get("registration"), dict) else {}
    if registration:
        print(
            "Registration: "
            f"{registration.get('status', 'unknown')} via {registration.get('applied_strategy', registration.get('runtime_mode', 'unknown'))}"
        )
    if details.get("failure_stage") == "registration":
        rejection_reason = registration.get("rejection_reason") or "Registration failed before part-level inspection could be trusted."
        print(f"Registration rejection: {rejection_reason}")
        for failure in registration.get("quality_gate_failures", [])[:2]:
            summary = failure.get("summary")
            if summary:
                print(f"Registration gate: {summary}")
    if details.get("reference_label"):
        print(
            f"Selected reference: {details.get('reference_label')}"
            f" ({details.get('reference_role', 'candidate')})"
        )
    print(f"ROI: {details['roi']}")
    print(f"Best angle correction: {details.get('best_angle_deg', 0.0):.2f} deg")
    print(f"Best shift correction: x={details.get('best_shift_x', 0)}, y={details.get('best_shift_y', 0)} px")
    print(f"Required coverage: {details['required_coverage']:.4f} (min {details['min_required_coverage']:.4f})")
    print(f"Outside allowed ratio: {details['outside_allowed_ratio']:.4f} (max {details['max_outside_allowed_ratio']:.4f})")
    print(f"Min section coverage: {details['min_section_coverage']:.4f} (min {details['min_section_coverage_limit']:.4f})")
    if details.get("worst_section_edge_distance_px") is not None:
        section_edge_gate_active = bool(details.get("section_edge_gate_active", False))
        if details.get("max_section_edge_distance_px") is not None:
            suffix = " [gate]" if section_edge_gate_active else " [info]"
            print(
                "Worst section edge distance: "
                f"{details['worst_section_edge_distance_px']:.3f}px "
                f"(max {details['max_section_edge_distance_px']:.3f}px){suffix}"
            )
        else:
            print(f"Worst section edge distance: {details['worst_section_edge_distance_px']:.3f}px")
    if details.get("worst_section_width_delta_ratio") is not None:
        section_width_gate_active = bool(details.get("section_width_gate_active", False))
        if details.get("max_section_width_delta_ratio") is not None:
            suffix = " [gate]" if section_width_gate_active else " [info]"
            print(
                "Worst section width drift: "
                f"{details['worst_section_width_delta_ratio']:.1%} "
                f"(max {details['max_section_width_delta_ratio']:.1%}){suffix}"
            )
        else:
            print(f"Worst section width drift: {details['worst_section_width_delta_ratio']:.1%}")
    if details.get("worst_section_center_offset_px") is not None:
        section_center_gate_active = bool(details.get("section_center_gate_active", False))
        if details.get("max_section_center_offset_px") is not None:
            suffix = " [gate]" if section_center_gate_active else " [info]"
            print(
                "Worst section center offset: "
                f"{details['worst_section_center_offset_px']:.3f}px "
                f"(max {details['max_section_center_offset_px']:.3f}px){suffix}"
            )
        else:
            print(f"Worst section center offset: {details['worst_section_center_offset_px']:.3f}px")
    if details.get("mean_edge_distance_px") is not None:
        edge_distance_gate_active = bool(details.get("edge_distance_gate_active", False))
        if details.get("max_mean_edge_distance_px") is not None:
            suffix = " [gate]" if edge_distance_gate_active else " [info]"
            print(
                "Mean edge distance: "
                f"{details['mean_edge_distance_px']:.3f}px "
                f"(max {details['max_mean_edge_distance_px']:.3f}px){suffix}"
            )
        else:
            print(f"Mean edge distance: {details['mean_edge_distance_px']:.3f}px")
    print(f"Sample white pixels: {details['sample_white_pixels']}")
    if details.get("section_coverages"):
        print("Section coverages:", ", ".join(f"{v:.3f}" for v in details["section_coverages"]))
    if details.get("section_edge_distances_px"):
        print("Section edge distances:", ", ".join(f"{v:.3f}" for v in details["section_edge_distances_px"]))
    if details.get("section_width_ratios"):
        print("Section width ratios:", ", ".join(f"{v:.3f}x" for v in details["section_width_ratios"]))
    if details.get("section_center_offsets_px"):
        print("Section center offsets:", ", ".join(f"{v:.3f}px" for v in details["section_center_offsets_px"]))
    if "ssim" in details:
        ssim_gate_active = bool(details.get("ssim_gate_active", False))
        if details.get("min_ssim") is not None:
            suffix = " [gate]" if ssim_gate_active else " [info]"
            print(f"SSIM: {details['ssim']:.4f} (min {details['min_ssim']:.4f}){suffix}")
        else:
            print(f"SSIM: {details['ssim']:.4f}")
    if "histogram_similarity" in details:
        print(f"Histogram similarity: {details['histogram_similarity']:.4f}")
    if "mse" in details:
        mse_gate_active = bool(details.get("mse_gate_active", False))
        if details.get("max_mse") is not None:
            suffix = " [gate]" if mse_gate_active else " [info]"
            print(f"MSE: {details['mse']:.2f} (max {details['max_mse']:.2f}){suffix}")
        else:
            print(f"MSE: {details['mse']:.2f}")
    if "anomaly_score" in details:
        anomaly_score = details.get("anomaly_score")
        anomaly_gate_active = bool(details.get("anomaly_gate_active", False))
        if anomaly_score is not None and details.get("min_anomaly_score") is not None:
            suffix = " [gate]" if anomaly_gate_active else " [info]"
            print(f"Anomaly score: {anomaly_score:.4f} (min {details['min_anomaly_score']:.4f}){suffix}")
        elif anomaly_score is not None:
            print(f"Anomaly score: {anomaly_score:.4f}")
    if details.get("debug_paths"):
        for key, path in details["debug_paths"].items():
            print(f"Debug {key}: {path}")


def run_capture_only(config: dict) -> int:
    result_code, image_path, stderr_text = capture_to_temp(config)
    if result_code != 0:
        print("Capture failed.")
        if stderr_text:
            print(stderr_text)
        cleanup_temp_image()
        return result_code

    print("Temporary capture completed.")
    cleanup_temp_image()
    return 0


def run_capture_and_inspect(config: dict, indicator) -> int:
    result_code, image_path, stderr_text = capture_to_temp(config)
    if result_code != 0:
        print("Capture failed.")
        if stderr_text:
            print(stderr_text)
        indicator.pulse_fail()
        cleanup_temp_image()
        return result_code

    try:
        runtime_context = build_inspection_runtime_context(
            config,
            active_paths_loader=get_active_runtime_paths,
            reference_candidates_loader=list_runtime_reference_candidates,
            anomaly_detector_loader=load_anomaly_detector,
        )
        for warning in get_inspection_runtime_warnings(config, runtime_context.anomaly_detector, runtime_context.active_paths):
            print(f"Warning: {warning}")
        if not runtime_context.reference_candidates:
            print("No active runtime references are available. Capture a golden reference first.")
            indicator.pulse_fail()
            return 1
        passed, details = inspect_against_references(
            config,
            image_path,
            runtime_context.reference_candidates,
            make_binary_mask,
            align_sample_mask,
            build_reference_regions,
            compute_section_masks,
            score_sample,
            evaluate_metrics,
            save_debug_outputs,
            import_cv2_and_numpy,
            dilate_mask,
            erode_mask,
            anomaly_detector=runtime_context.anomaly_detector,
        )
        print_inspection_result(passed, details)
        if passed:
            indicator.pulse_pass()
            return 0

        indicator.pulse_fail()
        return 1
    finally:
        cleanup_temp_image()

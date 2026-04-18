#!/usr/bin/env python3
"""Shared helpers for staged training asset lifecycle management."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def _call_with_optional_active_paths(action: Callable, target_id: str, active_paths: dict | None, *, state: str | None = None):
    if active_paths is None:
        if state is None:
            return action(target_id)
        return action(target_id, state=state)
    if state is None:
        return action(target_id, active_paths=active_paths)
    return action(target_id, active_paths=active_paths, state=state)


def stage_training_assets(
    record: dict,
    current_config: dict,
    *,
    final_class: str | None,
    image_path: Path | None,
    record_label_index: int,
    reference_strategy: str,
    active_paths: dict | None,
    stage_reference_candidate: Callable,
    stage_anomaly_training_sample: Callable,
) -> None:
    if final_class != "good" or image_path is None:
        return

    label_suffix = str(record_label_index)
    if reference_strategy in {"hybrid", "multi_good_experimental"}:
        stage_kwargs = {
            "label": f"Approved Good {label_suffix}",
            "source_record_id": record["record_id"],
        }
        if active_paths is not None:
            stage_kwargs["active_paths"] = active_paths
        staged_ok, staged_result = stage_reference_candidate(current_config, image_path, **stage_kwargs)
        if staged_ok:
            record["reference_candidate_id"] = staged_result["reference_id"]
            record["reference_candidate_state"] = staged_result["state"]

    sample_kwargs = {
        "label": f"Approved Good Sample {label_suffix}",
        "source_record_id": record["record_id"],
    }
    if active_paths is not None:
        sample_kwargs["active_paths"] = active_paths
    sample_ok, sample_result = stage_anomaly_training_sample(current_config, image_path, **sample_kwargs)
    if sample_ok:
        record["anomaly_sample_id"] = sample_result["sample_id"]
        record["anomaly_sample_state"] = sample_result["state"]


def commit_pending_training_records(
    training_data: list[dict],
    *,
    active_paths: dict | None,
    resolve_learning_class: Callable[[dict], str | None],
    activate_reference_candidate: Callable,
    activate_anomaly_training_sample: Callable,
) -> int:
    updated = 0
    for record in training_data:
        if record.get("learning_state", "committed") != "pending":
            continue

        candidate_id = record.get("reference_candidate_id")
        candidate_state = record.get("reference_candidate_state")
        learning_class = resolve_learning_class(record)
        if candidate_id and candidate_state == "pending" and learning_class == "good":
            activated = _call_with_optional_active_paths(
                activate_reference_candidate,
                candidate_id,
                active_paths,
            )
            if not activated:
                continue
            record["reference_candidate_state"] = "active"

        sample_id = record.get("anomaly_sample_id")
        sample_state = record.get("anomaly_sample_state")
        if sample_id and sample_state == "pending" and learning_class == "good":
            activated_sample = _call_with_optional_active_paths(
                activate_anomaly_training_sample,
                sample_id,
                active_paths,
            )
            if activated_sample:
                record["anomaly_sample_state"] = "active"

        record["learning_state"] = "committed"
        updated += 1
    return updated


def discard_pending_training_records(
    training_data: list[dict],
    *,
    active_paths: dict | None,
    discard_reference_candidate: Callable,
    discard_anomaly_training_sample: Callable,
) -> int:
    updated = 0
    for record in training_data:
        if record.get("learning_state", "committed") != "pending":
            continue

        candidate_id = record.get("reference_candidate_id")
        candidate_state = record.get("reference_candidate_state")
        if candidate_id and candidate_state == "pending":
            _call_with_optional_active_paths(
                discard_reference_candidate,
                candidate_id,
                active_paths,
                state="pending",
            )
            record["reference_candidate_state"] = "discarded"

        sample_id = record.get("anomaly_sample_id")
        sample_state = record.get("anomaly_sample_state")
        if sample_id and sample_state == "pending":
            _call_with_optional_active_paths(
                discard_anomaly_training_sample,
                sample_id,
                active_paths,
                state="pending",
            )
            record["anomaly_sample_state"] = "discarded"

        record["learning_state"] = "discarded"
        updated += 1
    return updated
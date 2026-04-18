#!/usr/bin/env python3
"""Shared runtime context helpers for inspection-oriented flows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class InspectionRuntimeContext:
    config: dict
    active_paths: dict
    reference_candidates: list
    anomaly_detector: object | None


def build_inspection_runtime_context(
    config: dict,
    *,
    active_paths_loader: Callable[[], dict],
    reference_candidates_loader: Callable[[dict, dict], list],
    anomaly_detector_loader: Callable[[dict], object | None],
) -> InspectionRuntimeContext:
    active_paths = active_paths_loader()
    return InspectionRuntimeContext(
        config=config,
        active_paths=active_paths,
        reference_candidates=reference_candidates_loader(config, active_paths),
        anomaly_detector=anomaly_detector_loader(active_paths),
    )


def refresh_inspection_runtime_context(
    context: InspectionRuntimeContext,
    *,
    config: dict | None,
    active_paths_loader: Callable[[], dict],
    reference_candidates_loader: Callable[[dict, dict], list],
    anomaly_detector_loader: Callable[[dict], object | None],
) -> InspectionRuntimeContext:
    next_config = config if config is not None else context.config
    context.config = next_config
    context.active_paths = active_paths_loader()
    context.reference_candidates = reference_candidates_loader(next_config, context.active_paths)
    context.anomaly_detector = anomaly_detector_loader(context.active_paths)
    return context
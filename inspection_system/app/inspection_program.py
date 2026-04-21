from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class InspectionLaneDefinition:
    lane_id: str
    lane_type: str
    authoritative: bool
    enabled: bool
    inspection_overrides: dict


@dataclass(frozen=True)
class InspectionProgramDefinition:
    program_id: str
    aggregation_policy: str
    lanes: tuple[InspectionLaneDefinition, ...]


def _normalize_lane(raw_lane: dict, index: int) -> InspectionLaneDefinition | None:
    if not isinstance(raw_lane, dict):
        return None

    lane_id = str(raw_lane.get("lane_id") or f"lane_{index + 1}").strip() or f"lane_{index + 1}"
    lane_type = str(raw_lane.get("lane_type") or "measurement").strip().lower() or "measurement"
    enabled = bool(raw_lane.get("enabled", True))
    authoritative = bool(raw_lane.get("authoritative", True))
    inspection_overrides = raw_lane.get("inspection")
    if not isinstance(inspection_overrides, dict):
        inspection_overrides = raw_lane.get("inspection_overrides")
    if not isinstance(inspection_overrides, dict):
        inspection_overrides = {}

    return InspectionLaneDefinition(
        lane_id=lane_id,
        lane_type=lane_type,
        authoritative=authoritative,
        enabled=enabled,
        inspection_overrides=deepcopy(inspection_overrides),
    )


def resolve_inspection_program(config: dict) -> InspectionProgramDefinition:
    inspection_cfg = config.get("inspection", {}) if isinstance(config, dict) else {}
    raw_program = config.get("inspection_program") if isinstance(config, dict) else None
    if not isinstance(raw_program, dict):
        raw_program = inspection_cfg.get("inspection_program") if isinstance(inspection_cfg, dict) else None
    if not isinstance(raw_program, dict):
        raw_program = {}

    raw_lanes = raw_program.get("lanes")
    if not isinstance(raw_lanes, list):
        raw_lanes = inspection_cfg.get("lanes") if isinstance(inspection_cfg, dict) else None
    if not isinstance(raw_lanes, list):
        raw_lanes = []

    lanes: list[InspectionLaneDefinition] = []
    for index, raw_lane in enumerate(raw_lanes):
        lane = _normalize_lane(raw_lane, index)
        if lane is not None and lane.enabled:
            lanes.append(lane)

    if not lanes:
        lanes = [
            InspectionLaneDefinition(
                lane_id="primary",
                lane_type="measurement",
                authoritative=True,
                enabled=True,
                inspection_overrides={},
            )
        ]

    program_id = str(raw_program.get("program_id") or inspection_cfg.get("program_id") or "default_program").strip()
    if not program_id:
        program_id = "default_program"

    aggregation_policy = str(raw_program.get("aggregation_policy") or "all_authoritative").strip().lower()
    if aggregation_policy not in {"all_authoritative"}:
        aggregation_policy = "all_authoritative"

    return InspectionProgramDefinition(
        program_id=program_id,
        aggregation_policy=aggregation_policy,
        lanes=tuple(lanes),
    )


def merge_lane_inspection_config(base_inspection_cfg: dict, lane: InspectionLaneDefinition) -> dict:
    merged = deepcopy(base_inspection_cfg if isinstance(base_inspection_cfg, dict) else {})
    merged.update(deepcopy(lane.inspection_overrides))
    merged["lane_id"] = lane.lane_id
    merged["lane_type"] = lane.lane_type
    merged["lane_authoritative"] = lane.authoritative
    return merged


def get_primary_lane(program: InspectionProgramDefinition) -> InspectionLaneDefinition:
    for lane in program.lanes:
        if lane.authoritative:
            return lane
    return program.lanes[0]


def aggregate_lane_results(program: InspectionProgramDefinition, lane_results: list[dict]) -> dict:
    if not lane_results:
        raise ValueError("At least one lane result is required to aggregate an inspection program.")

    primary_lane = get_primary_lane(program)
    primary_lane_id = primary_lane.lane_id
    primary_result = next((result for result in lane_results if result.get("lane_id") == primary_lane_id), lane_results[0])

    authoritative_results = [result for result in lane_results if bool(result.get("authoritative", False))]
    if not authoritative_results:
        authoritative_results = list(lane_results)

    failed_authoritative_results = [result for result in authoritative_results if not bool(result.get("passed", False))]
    failed_advisory_results = [
        result for result in lane_results if not bool(result.get("authoritative", False)) and not bool(result.get("passed", False))
    ]

    active_result = failed_authoritative_results[0] if failed_authoritative_results else primary_result
    failed_lane_ids = [str(result.get("lane_id")) for result in lane_results if not bool(result.get("passed", False))]
    failed_authoritative_lane_ids = [str(result.get("lane_id")) for result in failed_authoritative_results]
    failed_advisory_lane_ids = [str(result.get("lane_id")) for result in failed_advisory_results]

    return {
        "passed": len(failed_authoritative_results) == 0,
        "primary_lane_id": primary_lane_id,
        "active_lane_id": str(active_result.get("lane_id")),
        "failed_lane_ids": failed_lane_ids,
        "failed_authoritative_lane_ids": failed_authoritative_lane_ids,
        "failed_advisory_lane_ids": failed_advisory_lane_ids,
    }
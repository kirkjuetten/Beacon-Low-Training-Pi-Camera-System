from __future__ import annotations

from copy import deepcopy

from inspection_system.app.inspection_program import InspectionLaneDefinition, merge_lane_inspection_config


def execute_measurement_lane(
    lane: InspectionLaneDefinition,
    *,
    base_inspection_cfg: dict,
    measure_lane,
) -> dict:
    lane_inspection_cfg = merge_lane_inspection_config(base_inspection_cfg, lane)
    measurement_result = measure_lane(lane_inspection_cfg)
    return {
        "lane_id": lane.lane_id,
        "lane_type": lane.lane_type,
        "authoritative": lane.authoritative,
        "passed": bool(measurement_result.get("passed", False)),
        "inspection_cfg": lane_inspection_cfg,
        "measurement_result": measurement_result,
        "threshold_summary": deepcopy(measurement_result.get("threshold_summary", {})),
        "feature_measurements": deepcopy(measurement_result.get("feature_measurements", [])),
        "feature_position_summary": deepcopy(measurement_result.get("feature_position_summary")),
        "edge_measurement_frame": measurement_result.get("edge_measurement_frame"),
        "section_measurement_frame": measurement_result.get("section_measurement_frame"),
    }
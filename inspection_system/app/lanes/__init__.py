from __future__ import annotations

from inspection_system.app.lanes.measurement_lane import execute_measurement_lane


LANE_RUNNERS = {
    "measurement": execute_measurement_lane,
}


def execute_inspection_lane(lane, *, base_inspection_cfg: dict, measure_lane):
    runner = LANE_RUNNERS.get(str(lane.lane_type).strip().lower())
    if runner is None:
        raise ValueError(f"Unsupported inspection lane type: {lane.lane_type}")
    return runner(
        lane,
        base_inspection_cfg=base_inspection_cfg,
        measure_lane=measure_lane,
    )


__all__ = ["execute_inspection_lane"]
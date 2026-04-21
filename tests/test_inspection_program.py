from __future__ import annotations

from inspection_system.app.inspection_program import aggregate_lane_results, resolve_inspection_program


def test_resolve_inspection_program_defaults_to_single_primary_lane() -> None:
    program = resolve_inspection_program({"inspection": {}})

    assert program.program_id == "default_program"
    assert program.aggregation_policy == "all_authoritative"
    assert len(program.lanes) == 1
    assert program.lanes[0].lane_id == "primary"
    assert program.lanes[0].authoritative is True


def test_resolve_inspection_program_uses_explicit_lane_definitions() -> None:
    program = resolve_inspection_program(
        {
            "inspection_program": {
                "program_id": "part_program_a",
                "aggregation_policy": "all_authoritative",
                "lanes": [
                    {
                        "lane_id": "geometry",
                        "lane_type": "measurement",
                        "authoritative": True,
                        "inspection": {"lane_tag": "geometry"},
                    },
                    {
                        "lane_id": "print",
                        "lane_type": "measurement",
                        "authoritative": False,
                        "inspection": {"lane_tag": "print"},
                    },
                ],
            }
        }
    )

    assert program.program_id == "part_program_a"
    assert [lane.lane_id for lane in program.lanes] == ["geometry", "print"]
    assert program.lanes[1].authoritative is False
    assert program.lanes[0].inspection_overrides["lane_tag"] == "geometry"


def test_aggregate_lane_results_ignores_failed_advisory_lanes_for_program_pass() -> None:
    program = resolve_inspection_program(
        {
            "inspection_program": {
                "lanes": [
                    {"lane_id": "geometry", "authoritative": True},
                    {"lane_id": "print", "authoritative": False},
                ]
            }
        }
    )

    aggregation = aggregate_lane_results(
        program,
        [
            {"lane_id": "geometry", "authoritative": True, "passed": True},
            {"lane_id": "print", "authoritative": False, "passed": False},
        ],
    )

    assert aggregation["passed"] is True
    assert aggregation["primary_lane_id"] == "geometry"
    assert aggregation["failed_lane_ids"] == ["print"]
    assert aggregation["failed_authoritative_lane_ids"] == []
    assert aggregation["failed_advisory_lane_ids"] == ["print"]
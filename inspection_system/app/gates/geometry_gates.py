from __future__ import annotations


def _evaluate_maximum_gate(observed_value, threshold) -> tuple[bool, bool]:
    if threshold is None:
        return False, True
    return True, observed_value is not None and observed_value <= threshold


def evaluate_geometry_gates(
    *,
    mean_edge_distance_px,
    worst_section_edge_distance_px,
    worst_section_width_delta_ratio,
    worst_section_center_offset_px,
    max_mean_edge_distance_px,
    max_section_edge_distance_px,
    max_section_width_delta_ratio,
    max_section_center_offset_px,
    effective_max_mean_edge_distance_px,
    effective_max_section_edge_distance_px,
    effective_max_section_width_delta_ratio,
    effective_max_section_center_offset_px,
) -> dict:
    edge_distance_gate_active, edge_distance_pass = _evaluate_maximum_gate(
        mean_edge_distance_px,
        effective_max_mean_edge_distance_px,
    )
    section_edge_gate_active, section_edge_pass = _evaluate_maximum_gate(
        worst_section_edge_distance_px,
        effective_max_section_edge_distance_px,
    )
    section_width_gate_active, section_width_pass = _evaluate_maximum_gate(
        worst_section_width_delta_ratio,
        effective_max_section_width_delta_ratio,
    )
    section_center_gate_active, section_center_pass = _evaluate_maximum_gate(
        worst_section_center_offset_px,
        effective_max_section_center_offset_px,
    )

    return {
        "passed": (
            edge_distance_pass
            and section_edge_pass
            and section_width_pass
            and section_center_pass
        ),
        "summary": {
            "mean_edge_distance_px": mean_edge_distance_px,
            "max_mean_edge_distance_px": max_mean_edge_distance_px,
            "effective_max_mean_edge_distance_px": effective_max_mean_edge_distance_px,
            "worst_section_edge_distance_px": worst_section_edge_distance_px,
            "max_section_edge_distance_px": max_section_edge_distance_px,
            "effective_max_section_edge_distance_px": effective_max_section_edge_distance_px,
            "worst_section_width_delta_ratio": worst_section_width_delta_ratio,
            "max_section_width_delta_ratio": max_section_width_delta_ratio,
            "effective_max_section_width_delta_ratio": effective_max_section_width_delta_ratio,
            "worst_section_center_offset_px": worst_section_center_offset_px,
            "max_section_center_offset_px": max_section_center_offset_px,
            "effective_max_section_center_offset_px": effective_max_section_center_offset_px,
            "edge_distance_gate_active": edge_distance_gate_active,
            "section_edge_gate_active": section_edge_gate_active,
            "section_width_gate_active": section_width_gate_active,
            "section_center_gate_active": section_center_gate_active,
        },
    }
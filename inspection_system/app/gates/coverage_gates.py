from __future__ import annotations


def evaluate_coverage_gates(
    *,
    required_coverage: float,
    outside_allowed_ratio: float,
    min_section_coverage: float,
    min_required_coverage: float,
    max_outside_allowed_ratio: float,
    min_section_coverage_limit: float,
    effective_min_required_coverage: float,
    effective_max_outside_allowed_ratio: float,
    effective_min_section_coverage: float,
) -> dict:
    passed = (
        required_coverage >= effective_min_required_coverage
        and outside_allowed_ratio <= effective_max_outside_allowed_ratio
        and min_section_coverage >= effective_min_section_coverage
    )

    return {
        "passed": passed,
        "summary": {
            "required_coverage": required_coverage,
            "outside_allowed_ratio": outside_allowed_ratio,
            "min_section_coverage": min_section_coverage,
            "min_required_coverage": min_required_coverage,
            "max_outside_allowed_ratio": max_outside_allowed_ratio,
            "min_section_coverage_limit": min_section_coverage_limit,
            "effective_min_required_coverage": effective_min_required_coverage,
            "effective_max_outside_allowed_ratio": effective_max_outside_allowed_ratio,
            "effective_min_section_coverage": effective_min_section_coverage,
        },
    }
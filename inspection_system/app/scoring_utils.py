from __future__ import annotations


def score_sample(reference_allowed, reference_required, sample_mask, section_masks):
    sample_white = sample_mask > 0
    allowed_white = reference_allowed > 0
    required_white = reference_required > 0

    sample_count = int(sample_white.sum())
    required_count = int(required_white.sum())
    covered_required = int((sample_white & required_white).sum())
    outside_allowed = int((sample_white & (~allowed_white)).sum())

    required_coverage = covered_required / required_count if required_count else 0.0
    outside_allowed_ratio = outside_allowed / sample_count if sample_count else 1.0

    section_coverages = []
    for section in section_masks:
        section_white = section > 0
        denom = int(section_white.sum())
        if denom == 0:
            continue
        covered = int((sample_white & section_white).sum())
        section_coverages.append(covered / denom)

    min_section_coverage = min(section_coverages) if section_coverages else 0.0

    missing_required_mask = required_white & (~sample_white)
    outside_allowed_mask = sample_white & (~allowed_white)

    return {
        "required_coverage": required_coverage,
        "outside_allowed_ratio": outside_allowed_ratio,
        "min_section_coverage": min_section_coverage,
        "section_coverages": section_coverages,
        "sample_white_pixels": sample_count,
        "missing_required_mask": missing_required_mask,
        "outside_allowed_mask": outside_allowed_mask,
    }


def evaluate_metrics(metrics: dict, inspection_cfg: dict) -> tuple[bool, dict]:
    required_coverage = float(metrics["required_coverage"])
    outside_allowed_ratio = float(metrics["outside_allowed_ratio"])
    min_section_coverage = float(metrics["min_section_coverage"])

    min_required_coverage = float(inspection_cfg.get("min_required_coverage", 0.92))
    max_outside_allowed_ratio = float(inspection_cfg.get("max_outside_allowed_ratio", 0.02))
    min_section_coverage_limit = float(inspection_cfg.get("min_section_coverage", 0.85))

    passed = (
        required_coverage >= min_required_coverage
        and outside_allowed_ratio <= max_outside_allowed_ratio
        and min_section_coverage >= min_section_coverage_limit
    )

    return passed, {
        "required_coverage": required_coverage,
        "outside_allowed_ratio": outside_allowed_ratio,
        "min_section_coverage": min_section_coverage,
        "min_required_coverage": min_required_coverage,
        "max_outside_allowed_ratio": max_outside_allowed_ratio,
        "min_section_coverage_limit": min_section_coverage_limit,
    }

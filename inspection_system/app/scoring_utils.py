from __future__ import annotations


INSPECTION_MODE_GATES = {
    "mask_only": frozenset(),
    "mask_and_ssim": frozenset({"ssim", "mse"}),
    "mask_and_ml": frozenset({"anomaly"}),
    "full": frozenset({"ssim", "mse", "anomaly"}),
}

BLEND_MODE_WEIGHTS = {
    "hard_only": 0.0,
    "blend_conservative": 0.35,
    "blend_balanced": 0.55,
    "blend_aggressive": 0.75,
}

TOLERANCE_MODE_MULTIPLIERS = {
    "strict": 0.75,
    "balanced": 1.0,
    "forgiving": 1.35,
    "custom": 1.0,
}

LEARNED_RANGE_DIRECTIONS = {
    "required_coverage": "higher_is_better",
    "outside_allowed_ratio": "lower_is_better",
    "min_section_coverage": "higher_is_better",
    "mean_edge_distance_px": "lower_is_better",
    "worst_section_edge_distance_px": "lower_is_better",
    "ssim": "higher_is_better",
    "mse": "lower_is_better",
    "anomaly_score": "higher_is_better",
}

DEFAULT_LEARNED_MARGINS = {
    "required_coverage": 0.02,
    "outside_allowed_ratio": 0.01,
    "min_section_coverage": 0.02,
    "mean_edge_distance_px": 0.5,
    "worst_section_edge_distance_px": 0.5,
    "ssim": 0.02,
    "mse": 1.0,
    "anomaly_score": 0.05,
}


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_inspection_mode(value) -> str:
    normalized = str(value or "mask_only").strip().lower()
    if normalized not in INSPECTION_MODE_GATES:
        return "mask_only"
    return normalized


def normalize_blend_mode(value) -> str:
    normalized = str(value or "hard_only").strip().lower()
    if normalized not in BLEND_MODE_WEIGHTS:
        return "hard_only"
    return normalized


def normalize_tolerance_mode(value) -> str:
    normalized = str(value or "balanced").strip().lower()
    if normalized not in TOLERANCE_MODE_MULTIPLIERS:
        return "balanced"
    return normalized


def _get_learned_metric_range(inspection_cfg: dict, metric_name: str) -> dict | None:
    learned_ranges = inspection_cfg.get("learned_ranges", {})
    learned = learned_ranges.get(metric_name)
    if not isinstance(learned, dict):
        return None
    if "good_min" not in learned or "good_max" not in learned:
        return None
    return learned


def _get_learned_metric_threshold(metric_name: str, learned_range: dict | None, tolerance_mode: str):
    if learned_range is None:
        return None

    direction = LEARNED_RANGE_DIRECTIONS[metric_name]
    good_min = _optional_float(learned_range.get("good_min"))
    good_max = _optional_float(learned_range.get("good_max"))
    if good_min is None or good_max is None:
        return None

    span = max(0.0, good_max - good_min)
    tolerance_multiplier = TOLERANCE_MODE_MULTIPLIERS[tolerance_mode]
    margin = max(DEFAULT_LEARNED_MARGINS[metric_name], span * 0.2 * tolerance_multiplier)

    if direction == "higher_is_better":
        return max(0.0, good_min - margin)
    return good_max + margin


def _blend_threshold(metric_name: str, configured_threshold, learned_threshold, blend_mode: str):
    if learned_threshold is None:
        return configured_threshold
    weight = BLEND_MODE_WEIGHTS[blend_mode]
    if configured_threshold is None or weight <= 0.0:
        return learned_threshold if configured_threshold is None and weight > 0.0 else configured_threshold

    blended = (1.0 - weight) * configured_threshold + weight * learned_threshold
    direction = LEARNED_RANGE_DIRECTIONS[metric_name]
    if direction == "higher_is_better":
        return min(configured_threshold, blended)
    return max(configured_threshold, blended)


def resolve_inspection_mode_details(inspection_cfg: dict) -> dict:
    inspection_mode = normalize_inspection_mode(inspection_cfg.get("inspection_mode", "mask_only"))
    included_gates = INSPECTION_MODE_GATES[inspection_mode]

    max_mean_edge_distance_px = _optional_float(inspection_cfg.get("max_mean_edge_distance_px"))
    max_section_edge_distance_px = _optional_float(inspection_cfg.get("max_section_edge_distance_px"))
    min_ssim = _optional_float(inspection_cfg.get("min_ssim"))
    max_mse = _optional_float(inspection_cfg.get("max_mse"))
    min_anomaly_score = _optional_float(inspection_cfg.get("min_anomaly_score"))

    return {
        "inspection_mode": inspection_mode,
        "included_gates": included_gates,
        "max_mean_edge_distance_px": max_mean_edge_distance_px,
        "max_section_edge_distance_px": max_section_edge_distance_px,
        "min_ssim": min_ssim,
        "max_mse": max_mse,
        "min_anomaly_score": min_anomaly_score,
        "edge_distance_gate_active": max_mean_edge_distance_px is not None,
        "section_edge_gate_active": max_section_edge_distance_px is not None,
        "ssim_gate_active": "ssim" in included_gates and min_ssim is not None,
        "mse_gate_active": "mse" in included_gates and max_mse is not None,
        "anomaly_gate_active": "anomaly" in included_gates and min_anomaly_score is not None,
    }


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
    mean_edge_distance_px = _optional_float(metrics.get("mean_edge_distance_px"))
    worst_section_edge_distance_px = _optional_float(metrics.get("worst_section_edge_distance_px"))
    ssim_value = _optional_float(metrics.get("ssim"))
    mse_value = _optional_float(metrics.get("mse"))
    anomaly_score = _optional_float(metrics.get("anomaly_score"))

    min_required_coverage = float(inspection_cfg.get("min_required_coverage", 0.92))
    max_outside_allowed_ratio = float(inspection_cfg.get("max_outside_allowed_ratio", 0.02))
    min_section_coverage_limit = float(inspection_cfg.get("min_section_coverage", 0.85))

    mode_details = resolve_inspection_mode_details(inspection_cfg)
    inspection_mode = mode_details["inspection_mode"]
    configured_max_mean_edge_distance_px = mode_details["max_mean_edge_distance_px"]
    configured_max_section_edge_distance_px = mode_details["max_section_edge_distance_px"]
    configured_min_ssim = mode_details["min_ssim"]
    configured_max_mse = mode_details["max_mse"]
    configured_min_anomaly_score = mode_details["min_anomaly_score"]

    blend_mode = normalize_blend_mode(inspection_cfg.get("blend_mode", "hard_only"))
    tolerance_mode = normalize_tolerance_mode(inspection_cfg.get("tolerance_mode", "balanced"))

    learned_required = _get_learned_metric_threshold(
        "required_coverage",
        _get_learned_metric_range(inspection_cfg, "required_coverage"),
        tolerance_mode,
    )
    learned_outside = _get_learned_metric_threshold(
        "outside_allowed_ratio",
        _get_learned_metric_range(inspection_cfg, "outside_allowed_ratio"),
        tolerance_mode,
    )
    learned_section = _get_learned_metric_threshold(
        "min_section_coverage",
        _get_learned_metric_range(inspection_cfg, "min_section_coverage"),
        tolerance_mode,
    )
    learned_edge_distance = _get_learned_metric_threshold(
        "mean_edge_distance_px",
        _get_learned_metric_range(inspection_cfg, "mean_edge_distance_px"),
        tolerance_mode,
    )
    learned_section_edge_distance = _get_learned_metric_threshold(
        "worst_section_edge_distance_px",
        _get_learned_metric_range(inspection_cfg, "worst_section_edge_distance_px"),
        tolerance_mode,
    )
    learned_ssim = _get_learned_metric_threshold(
        "ssim",
        _get_learned_metric_range(inspection_cfg, "ssim"),
        tolerance_mode,
    )
    learned_mse = _get_learned_metric_threshold(
        "mse",
        _get_learned_metric_range(inspection_cfg, "mse"),
        tolerance_mode,
    )
    learned_anomaly = _get_learned_metric_threshold(
        "anomaly_score",
        _get_learned_metric_range(inspection_cfg, "anomaly_score"),
        tolerance_mode,
    )

    effective_min_required_coverage = _blend_threshold(
        "required_coverage", min_required_coverage, learned_required, blend_mode
    )
    effective_max_outside_allowed_ratio = _blend_threshold(
        "outside_allowed_ratio", max_outside_allowed_ratio, learned_outside, blend_mode
    )
    effective_min_section_coverage = _blend_threshold(
        "min_section_coverage", min_section_coverage_limit, learned_section, blend_mode
    )
    effective_max_mean_edge_distance_px = _blend_threshold(
        "mean_edge_distance_px",
        configured_max_mean_edge_distance_px,
        learned_edge_distance,
        blend_mode,
    )
    effective_max_section_edge_distance_px = _blend_threshold(
        "worst_section_edge_distance_px",
        configured_max_section_edge_distance_px,
        learned_section_edge_distance,
        blend_mode,
    )
    effective_min_ssim = _blend_threshold("ssim", configured_min_ssim, learned_ssim, blend_mode)
    effective_max_mse = _blend_threshold("mse", configured_max_mse, learned_mse, blend_mode)
    effective_min_anomaly_score = _blend_threshold(
        "anomaly_score", configured_min_anomaly_score, learned_anomaly, blend_mode
    )

    edge_distance_gate_active = effective_max_mean_edge_distance_px is not None
    section_edge_gate_active = effective_max_section_edge_distance_px is not None
    ssim_gate_active = "ssim" in mode_details["included_gates"] and effective_min_ssim is not None
    mse_gate_active = "mse" in mode_details["included_gates"] and effective_max_mse is not None
    anomaly_gate_active = "anomaly" in mode_details["included_gates"] and effective_min_anomaly_score is not None

    edge_distance_pass = True if not edge_distance_gate_active else (
        mean_edge_distance_px is not None and mean_edge_distance_px <= effective_max_mean_edge_distance_px
    )
    section_edge_pass = True if not section_edge_gate_active else (
        worst_section_edge_distance_px is not None
        and worst_section_edge_distance_px <= effective_max_section_edge_distance_px
    )
    ssim_pass = True if not ssim_gate_active else (ssim_value is not None and ssim_value >= effective_min_ssim)
    mse_pass = True if not mse_gate_active else (mse_value is not None and mse_value <= effective_max_mse)
    anomaly_pass = True if not anomaly_gate_active else (
        anomaly_score is not None and anomaly_score >= effective_min_anomaly_score
    )

    passed = (
        required_coverage >= effective_min_required_coverage
        and outside_allowed_ratio <= effective_max_outside_allowed_ratio
        and min_section_coverage >= effective_min_section_coverage
        and edge_distance_pass
        and section_edge_pass
        and ssim_pass
        and mse_pass
        and anomaly_pass
    )

    return passed, {
        "required_coverage": required_coverage,
        "outside_allowed_ratio": outside_allowed_ratio,
        "min_section_coverage": min_section_coverage,
        "min_required_coverage": min_required_coverage,
        "max_outside_allowed_ratio": max_outside_allowed_ratio,
        "min_section_coverage_limit": min_section_coverage_limit,
        "effective_min_required_coverage": effective_min_required_coverage,
        "effective_max_outside_allowed_ratio": effective_max_outside_allowed_ratio,
        "effective_min_section_coverage": effective_min_section_coverage,
        "mean_edge_distance_px": mean_edge_distance_px,
        "max_mean_edge_distance_px": configured_max_mean_edge_distance_px,
        "effective_max_mean_edge_distance_px": effective_max_mean_edge_distance_px,
        "worst_section_edge_distance_px": worst_section_edge_distance_px,
        "max_section_edge_distance_px": configured_max_section_edge_distance_px,
        "effective_max_section_edge_distance_px": effective_max_section_edge_distance_px,
        "ssim": ssim_value,
        "mse": mse_value,
        "anomaly_score": anomaly_score,
        "min_ssim": configured_min_ssim,
        "max_mse": configured_max_mse,
        "min_anomaly_score": configured_min_anomaly_score,
        "effective_min_ssim": effective_min_ssim,
        "effective_max_mse": effective_max_mse,
        "effective_min_anomaly_score": effective_min_anomaly_score,
        "inspection_mode": inspection_mode,
        "blend_mode": blend_mode,
        "tolerance_mode": tolerance_mode,
        "learned_ranges_active": blend_mode != "hard_only" and any(
            threshold is not None
            for threshold in [
                learned_required,
                learned_outside,
                learned_section,
                learned_edge_distance,
                learned_section_edge_distance,
                learned_ssim,
                learned_mse,
                learned_anomaly,
            ]
        ),
        "edge_distance_gate_active": edge_distance_gate_active,
        "section_edge_gate_active": section_edge_gate_active,
        "ssim_gate_active": ssim_gate_active,
        "mse_gate_active": mse_gate_active,
        "anomaly_gate_active": anomaly_gate_active,
    }

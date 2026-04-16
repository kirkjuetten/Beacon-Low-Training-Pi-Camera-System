from __future__ import annotations


INSPECTION_MODE_GATES = {
    "mask_only": frozenset(),
    "mask_and_ssim": frozenset({"ssim", "mse"}),
    "mask_and_ml": frozenset({"anomaly"}),
    "full": frozenset({"ssim", "mse", "anomaly"}),
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


def resolve_inspection_mode_details(inspection_cfg: dict) -> dict:
    inspection_mode = normalize_inspection_mode(inspection_cfg.get("inspection_mode", "mask_only"))
    included_gates = INSPECTION_MODE_GATES[inspection_mode]

    min_ssim = _optional_float(inspection_cfg.get("min_ssim"))
    max_mse = _optional_float(inspection_cfg.get("max_mse"))
    min_anomaly_score = _optional_float(inspection_cfg.get("min_anomaly_score"))

    return {
        "inspection_mode": inspection_mode,
        "included_gates": included_gates,
        "min_ssim": min_ssim,
        "max_mse": max_mse,
        "min_anomaly_score": min_anomaly_score,
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
    ssim_value = _optional_float(metrics.get("ssim"))
    mse_value = _optional_float(metrics.get("mse"))
    anomaly_score = _optional_float(metrics.get("anomaly_score"))

    min_required_coverage = float(inspection_cfg.get("min_required_coverage", 0.92))
    max_outside_allowed_ratio = float(inspection_cfg.get("max_outside_allowed_ratio", 0.02))
    min_section_coverage_limit = float(inspection_cfg.get("min_section_coverage", 0.85))

    mode_details = resolve_inspection_mode_details(inspection_cfg)
    inspection_mode = mode_details["inspection_mode"]
    min_ssim = mode_details["min_ssim"]
    max_mse = mode_details["max_mse"]
    min_anomaly_score = mode_details["min_anomaly_score"]
    ssim_gate_active = mode_details["ssim_gate_active"]
    mse_gate_active = mode_details["mse_gate_active"]
    anomaly_gate_active = mode_details["anomaly_gate_active"]

    ssim_pass = True if not ssim_gate_active else (ssim_value is not None and ssim_value >= min_ssim)
    mse_pass = True if not mse_gate_active else (mse_value is not None and mse_value <= max_mse)
    anomaly_pass = True if not anomaly_gate_active else (
        anomaly_score is not None and anomaly_score >= min_anomaly_score
    )

    passed = (
        required_coverage >= min_required_coverage
        and outside_allowed_ratio <= max_outside_allowed_ratio
        and min_section_coverage >= min_section_coverage_limit
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
        "ssim": ssim_value,
        "mse": mse_value,
        "anomaly_score": anomaly_score,
        "min_ssim": min_ssim,
        "max_mse": max_mse,
        "min_anomaly_score": min_anomaly_score,
        "inspection_mode": inspection_mode,
        "ssim_gate_active": ssim_gate_active,
        "mse_gate_active": mse_gate_active,
        "anomaly_gate_active": anomaly_gate_active,
    }

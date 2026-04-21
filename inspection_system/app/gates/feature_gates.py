from __future__ import annotations

from copy import deepcopy

from inspection_system.app.feature_measurement_utils import summarize_feature_measurements


FEATURE_GATE_KEYS = {
    "dx_px": "max_feature_dx_px",
    "dy_px": "max_feature_dy_px",
    "radial_offset_px": "max_feature_radial_offset_px",
    "pair_spacing_delta_px": "max_feature_pair_spacing_delta_px",
}

FEATURE_GATE_ALIASES = {
    "dx_px": "max_dx_px",
    "dy_px": "max_dy_px",
    "radial_offset_px": "max_radial_offset_px",
    "pair_spacing_delta_px": "max_pair_spacing_delta_px",
}


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if numeric_value != numeric_value or numeric_value in (float("inf"), float("-inf")):
        return None
    return numeric_value


def _first_configured_value(raw_thresholds: dict, inspection_cfg: dict, metric_name: str):
    for key in (FEATURE_GATE_ALIASES[metric_name], FEATURE_GATE_KEYS[metric_name]):
        value = _optional_float(raw_thresholds.get(key))
        if value is not None:
            return value
        value = _optional_float(inspection_cfg.get(key))
        if value is not None:
            return value
    return None


def _resolve_feature_gate_thresholds(inspection_cfg: dict) -> dict:
    raw_thresholds = inspection_cfg.get("feature_gate_thresholds", {})
    if not isinstance(raw_thresholds, dict):
        raw_thresholds = {}

    return {
        metric_name: _first_configured_value(raw_thresholds, inspection_cfg, metric_name)
        for metric_name in FEATURE_GATE_KEYS
    }


def _supports_feature_gate(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("feature_family") == "datum_section":
        return False
    return any(entry.get(metric_name) is not None for metric_name in FEATURE_GATE_KEYS)


def _observed_metric_value(entry: dict, metric_name: str):
    observed_value = _optional_float(entry.get(metric_name))
    if observed_value is None:
        return None
    if metric_name in {"dx_px", "dy_px", "pair_spacing_delta_px"}:
        return abs(observed_value)
    return observed_value


def _rank_failure(failure: dict) -> tuple[int, float, float]:
    if not failure.get("sample_detected", True):
        return (0, float("inf"), float("inf"))
    return (1, float(failure.get("ratio", float("inf"))), float(-failure.get("margin", float("inf"))))


def evaluate_feature_gates(feature_measurements: list[dict], inspection_cfg: dict) -> dict:
    thresholds = _resolve_feature_gate_thresholds(inspection_cfg)
    feature_gate_active = any(value is not None for value in thresholds.values())
    summary = {
        "feature_gate_active": False,
        "feature_gate_passed": True,
        "feature_gate_failed_check_count": 0,
        "feature_gate_metric": None,
        "feature_gate_feature_key": None,
        "feature_gate_failure_cause": None,
        "feature_gate_observed_value": None,
        "feature_gate_threshold": None,
        "feature_gate_margin_px": None,
        "feature_gate_ratio": None,
        "max_feature_dx_px": thresholds["dx_px"],
        "max_feature_dy_px": thresholds["dy_px"],
        "max_feature_radial_offset_px": thresholds["radial_offset_px"],
        "max_feature_pair_spacing_delta_px": thresholds["pair_spacing_delta_px"],
    }
    if not feature_gate_active:
        return {
            "passed": True,
            "summary": summary,
            "feature_position_summary": None,
        }

    eligible_measurements = [entry for entry in feature_measurements if _supports_feature_gate(entry)]
    if not eligible_measurements:
        return {
            "passed": True,
            "summary": summary,
            "feature_position_summary": None,
        }

    summary["feature_gate_active"] = True
    failures: list[dict] = []
    minimum_margin = None

    for entry in eligible_measurements:
        if not bool(entry.get("sample_detected", False)):
            failures.append(
                {
                    "entry": entry,
                    "metric": "sample_detected",
                    "observed_value": None,
                    "threshold": None,
                    "margin": -1.0,
                    "ratio": float("inf"),
                    "sample_detected": False,
                }
            )
            continue

        for metric_name, threshold in thresholds.items():
            if threshold is None:
                continue
            observed_value = _observed_metric_value(entry, metric_name)
            if observed_value is None:
                continue

            margin = float(threshold) - float(observed_value)
            ratio = float(observed_value) / float(threshold) if float(threshold) > 0 else 1.0
            if minimum_margin is None or margin < minimum_margin:
                minimum_margin = margin
            if margin < 0.0:
                failures.append(
                    {
                        "entry": entry,
                        "metric": metric_name,
                        "observed_value": observed_value,
                        "threshold": float(threshold),
                        "margin": margin,
                        "ratio": ratio,
                        "sample_detected": True,
                    }
                )

    if not failures:
        summary["feature_gate_margin_px"] = minimum_margin
        return {
            "passed": True,
            "summary": summary,
            "feature_position_summary": None,
        }

    selected_failure = max(failures, key=_rank_failure)
    selected_entry = deepcopy(selected_failure["entry"])
    selected_summary = summarize_feature_measurements([selected_entry]) or deepcopy(selected_entry)
    selected_summary["feature_count"] = len(eligible_measurements)

    summary.update(
        {
            "feature_gate_passed": False,
            "feature_gate_failed_check_count": len(failures),
            "feature_gate_metric": selected_failure["metric"],
            "feature_gate_feature_key": selected_entry.get("feature_key"),
            "feature_gate_failure_cause": selected_entry.get(
                "failure_cause",
                "feature_not_found" if not selected_failure.get("sample_detected", True) else "feature_position",
            ),
            "feature_gate_observed_value": selected_failure.get("observed_value"),
            "feature_gate_threshold": selected_failure.get("threshold"),
            "feature_gate_margin_px": selected_failure.get("margin"),
            "feature_gate_ratio": selected_failure.get("ratio"),
        }
    )

    return {
        "passed": False,
        "summary": summary,
        "feature_position_summary": selected_summary,
    }
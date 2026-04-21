from __future__ import annotations


def _evaluate_minimum_gate(observed_value, threshold) -> tuple[bool, bool]:
    if threshold is None:
        return False, True
    return True, observed_value is not None and observed_value >= threshold


def _evaluate_maximum_gate(observed_value, threshold) -> tuple[bool, bool]:
    if threshold is None:
        return False, True
    return True, observed_value is not None and observed_value <= threshold


def evaluate_anomaly_gates(
    *,
    inspection_mode: str,
    included_gates: frozenset[str],
    ssim_value,
    mse_value,
    anomaly_score,
    min_ssim,
    max_mse,
    min_anomaly_score,
    effective_min_ssim,
    effective_max_mse,
    effective_min_anomaly_score,
) -> dict:
    ssim_gate_active, ssim_pass = _evaluate_minimum_gate(
        ssim_value,
        effective_min_ssim if "ssim" in included_gates else None,
    )
    mse_gate_active, mse_pass = _evaluate_maximum_gate(
        mse_value,
        effective_max_mse if "mse" in included_gates else None,
    )
    anomaly_gate_active, anomaly_pass = _evaluate_minimum_gate(
        anomaly_score,
        effective_min_anomaly_score if "anomaly" in included_gates else None,
    )

    return {
        "passed": ssim_pass and mse_pass and anomaly_pass,
        "summary": {
            "ssim": ssim_value,
            "mse": mse_value,
            "anomaly_score": anomaly_score,
            "min_ssim": min_ssim,
            "max_mse": max_mse,
            "min_anomaly_score": min_anomaly_score,
            "effective_min_ssim": effective_min_ssim,
            "effective_max_mse": effective_max_mse,
            "effective_min_anomaly_score": effective_min_anomaly_score,
            "inspection_mode": inspection_mode,
            "ssim_gate_active": ssim_gate_active,
            "mse_gate_active": mse_gate_active,
            "anomaly_gate_active": anomaly_gate_active,
        },
    }
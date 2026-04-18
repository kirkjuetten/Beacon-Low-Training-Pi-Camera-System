#!/usr/bin/env python3
"""Shared production result interpretation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


GOOD = "good"
REJECT = "reject"
REVIEW = "review"

REASON_MISSING_PRINT = "missing_print"
REASON_EXTRA_PRINT = "extra_print"
REASON_UNEVEN_PRINT = "uneven_print"
REASON_REFERENCE_MISMATCH = "reference_mismatch"
REASON_REGISTRATION_FAILURE = "registration_failure"

REASON_ORDER = [
    REASON_MISSING_PRINT,
    REASON_EXTRA_PRINT,
    REASON_UNEVEN_PRINT,
    REASON_REFERENCE_MISMATCH,
    REASON_REGISTRATION_FAILURE,
]

REASON_LABELS = {
    REASON_MISSING_PRINT: "Missing Print",
    REASON_EXTRA_PRINT: "Extra Print",
    REASON_UNEVEN_PRINT: "Uneven Print",
    REASON_REFERENCE_MISMATCH: "Reference Mismatch",
    REASON_REGISTRATION_FAILURE: "Placement / Registration",
}


def _registration_rejection_reason(details: dict) -> Optional[str]:
    registration = details.get("registration", {})
    if isinstance(registration, dict) and registration.get("rejection_reason"):
        return str(registration["rejection_reason"])
    if details.get("registration_rejection_reason"):
        return str(details["registration_rejection_reason"])
    return None


@dataclass(frozen=True)
class ProductionOutcome:
    status: str
    banner_text: str
    primary_reason: Optional[str]
    summary_lines: list[str]


def _threshold_ratio(value: Optional[float], threshold: Optional[float], high_is_bad: bool) -> float:
    if value is None or threshold is None:
        return 0.0
    if threshold <= 0:
        return 1.0
    if high_is_bad:
        return threshold / max(value, 1e-9)
    return value / threshold


def _metric_exceeds_limit(details: dict, value_key: str, threshold_key: str, *, high_is_bad: bool = True) -> bool:
    value = details.get(value_key)
    threshold = details.get(threshold_key)
    if value is None or threshold is None:
        return True
    if high_is_bad:
        return float(value) > float(threshold)
    return float(value) < float(threshold)


def _failed_checks(details: dict) -> list[str]:
    if str(details.get("failure_stage", "")).lower() == "registration" or _registration_rejection_reason(details):
        return [REASON_REGISTRATION_FAILURE]

    failures: list[str] = []

    if float(details.get("required_coverage", 0.0)) < float(details.get("min_required_coverage", 0.0)):
        failures.append(REASON_MISSING_PRINT)
    if float(details.get("outside_allowed_ratio", 0.0)) > float(details.get("max_outside_allowed_ratio", 0.0)):
        failures.append(REASON_EXTRA_PRINT)

    min_section_limit = float(details.get("min_section_coverage_limit", 0.0))
    if min_section_limit > 0 and float(details.get("min_section_coverage", 0.0)) < min_section_limit:
        failures.append(REASON_UNEVEN_PRINT)

    mismatch_failure = False
    if bool(details.get("section_center_gate_active", False)):
        mismatch_failure = mismatch_failure or _metric_exceeds_limit(
            details,
            "worst_section_center_offset_px",
            "max_section_center_offset_px",
        )
    if bool(details.get("section_width_gate_active", False)):
        mismatch_failure = mismatch_failure or _metric_exceeds_limit(
            details,
            "worst_section_width_delta_ratio",
            "max_section_width_delta_ratio",
        )
    if bool(details.get("section_edge_gate_active", False)):
        mismatch_failure = mismatch_failure or _metric_exceeds_limit(
            details,
            "worst_section_edge_distance_px",
            "max_section_edge_distance_px",
        )
    if bool(details.get("ssim_gate_active", False)):
        mismatch_failure = mismatch_failure or _metric_exceeds_limit(
            details,
            "ssim",
            "min_ssim",
            high_is_bad=False,
        )
    if bool(details.get("mse_gate_active", False)):
        mismatch_failure = mismatch_failure or _metric_exceeds_limit(
            details,
            "mse",
            "max_mse",
        )
    if bool(details.get("edge_distance_gate_active", False)):
        mismatch_failure = mismatch_failure or _metric_exceeds_limit(
            details,
            "mean_edge_distance_px",
            "max_mean_edge_distance_px",
        )
    if bool(details.get("anomaly_gate_active", False)):
        mismatch_failure = mismatch_failure or _metric_exceeds_limit(
            details,
            "anomaly_score",
            "min_anomaly_score",
            high_is_bad=False,
        )

    if mismatch_failure:
        failures.append(REASON_REFERENCE_MISMATCH)

    return failures


def _is_borderline_failure(details: dict, failure: str) -> bool:
    if failure == REASON_MISSING_PRINT:
        return _threshold_ratio(
            float(details.get("required_coverage", 0.0)),
            float(details.get("min_required_coverage", 0.0)),
            high_is_bad=False,
        ) >= 0.9

    if failure == REASON_EXTRA_PRINT:
        return _threshold_ratio(
            float(details.get("outside_allowed_ratio", 0.0)),
            float(details.get("max_outside_allowed_ratio", 0.0)),
            high_is_bad=True,
        ) >= (1.0 / 1.1)

    if failure == REASON_UNEVEN_PRINT:
        return _threshold_ratio(
            float(details.get("min_section_coverage", 0.0)),
            float(details.get("min_section_coverage_limit", 0.0)),
            high_is_bad=False,
        ) >= 0.9

    if failure == REASON_REFERENCE_MISMATCH:
        if bool(details.get("section_center_gate_active", False)) and _metric_exceeds_limit(
            details,
            "worst_section_center_offset_px",
            "max_section_center_offset_px",
        ):
            return _threshold_ratio(
                details.get("worst_section_center_offset_px"),
                details.get("max_section_center_offset_px"),
                high_is_bad=True,
            ) >= (1.0 / 1.1)
        if bool(details.get("section_width_gate_active", False)) and _metric_exceeds_limit(
            details,
            "worst_section_width_delta_ratio",
            "max_section_width_delta_ratio",
        ):
            return _threshold_ratio(
                details.get("worst_section_width_delta_ratio"),
                details.get("max_section_width_delta_ratio"),
                high_is_bad=True,
            ) >= (1.0 / 1.1)
        if bool(details.get("section_edge_gate_active", False)) and _metric_exceeds_limit(
            details,
            "worst_section_edge_distance_px",
            "max_section_edge_distance_px",
        ):
            return _threshold_ratio(
                details.get("worst_section_edge_distance_px"),
                details.get("max_section_edge_distance_px"),
                high_is_bad=True,
            ) >= (1.0 / 1.1)
        if bool(details.get("edge_distance_gate_active", False)) and _metric_exceeds_limit(
            details,
            "mean_edge_distance_px",
            "max_mean_edge_distance_px",
        ):
            return _threshold_ratio(
                details.get("mean_edge_distance_px"),
                details.get("max_mean_edge_distance_px"),
                high_is_bad=True,
            ) >= (1.0 / 1.1)
        if bool(details.get("ssim_gate_active", False)) and _metric_exceeds_limit(
            details,
            "ssim",
            "min_ssim",
            high_is_bad=False,
        ):
            return _threshold_ratio(details.get("ssim"), details.get("min_ssim"), high_is_bad=False) >= 0.95
        if bool(details.get("mse_gate_active", False)) and _metric_exceeds_limit(
            details,
            "mse",
            "max_mse",
        ):
            return _threshold_ratio(details.get("mse"), details.get("max_mse"), high_is_bad=True) >= (1.0 / 1.1)
        if bool(details.get("anomaly_gate_active", False)) and _metric_exceeds_limit(
            details,
            "anomaly_score",
            "min_anomaly_score",
            high_is_bad=False,
        ):
            return _threshold_ratio(
                details.get("anomaly_score"),
                details.get("min_anomaly_score"),
                high_is_bad=False,
            ) >= 0.95
        return True

    if failure == REASON_REGISTRATION_FAILURE:
        return False

    return False


def _make_summary_lines(status: str, primary_reason: Optional[str], details: dict) -> list[str]:
    registration_reason = _registration_rejection_reason(details)
    required_coverage = float(details.get("required_coverage", 0.0))
    min_required_coverage = float(details.get("min_required_coverage", 0.0))
    outside_allowed_ratio = float(details.get("outside_allowed_ratio", 0.0))
    max_outside_allowed_ratio = float(details.get("max_outside_allowed_ratio", 0.0))
    min_section_coverage = float(details.get("min_section_coverage", 0.0))
    min_section_limit = float(details.get("min_section_coverage_limit", 0.0))

    if status == GOOD:
        return [
            "Approved part.",
            f"Coverage {required_coverage:.1%} / {min_required_coverage:.1%}",
            f"Outside print {outside_allowed_ratio:.1%} / {max_outside_allowed_ratio:.1%}",
        ]

    if primary_reason == REASON_REGISTRATION_FAILURE:
        lines = ["Part position could not be verified."]
        if registration_reason:
            lines.append(registration_reason)
        lines.append("Reload part and inspect again.")
        return lines[:3]

    if status == REVIEW:
        return [
            "Borderline part.",
            "Manual review required.",
            f"Place in red bin: {REASON_LABELS.get(primary_reason or REASON_REFERENCE_MISMATCH, 'Needs Review')}",
        ]

    lines = []
    if primary_reason == REASON_MISSING_PRINT:
        lines.append("Missing print area.")
        lines.append(f"Coverage {required_coverage:.1%} / {min_required_coverage:.1%}")
    elif primary_reason == REASON_EXTRA_PRINT:
        lines.append("Extra print outside allowed zone.")
        lines.append(f"Outside print {outside_allowed_ratio:.1%} / {max_outside_allowed_ratio:.1%}")
    elif primary_reason == REASON_UNEVEN_PRINT:
        lines.append("Print is uneven across the part.")
        lines.append(f"Weakest section {min_section_coverage:.1%} / {min_section_limit:.1%}")
    else:
        lines.append("Part does not match reference.")
        if (
            bool(details.get("section_center_gate_active", False))
            and _metric_exceeds_limit(details, "worst_section_center_offset_px", "max_section_center_offset_px")
            and details.get("worst_section_center_offset_px") is not None
            and details.get("max_section_center_offset_px") is not None
        ):
            lines.append(
                f"Section center offset {float(details['worst_section_center_offset_px']):.2f}px / {float(details['max_section_center_offset_px']):.2f}px"
            )
        elif (
            bool(details.get("section_width_gate_active", False))
            and _metric_exceeds_limit(details, "worst_section_width_delta_ratio", "max_section_width_delta_ratio")
            and details.get("worst_section_width_delta_ratio") is not None
            and details.get("max_section_width_delta_ratio") is not None
        ):
            lines.append(
                f"Section width drift {float(details['worst_section_width_delta_ratio']):.1%} / {float(details['max_section_width_delta_ratio']):.1%}"
            )
        elif (
            bool(details.get("section_edge_gate_active", False))
            and _metric_exceeds_limit(details, "worst_section_edge_distance_px", "max_section_edge_distance_px")
            and details.get("worst_section_edge_distance_px") is not None
            and details.get("max_section_edge_distance_px") is not None
        ):
            lines.append(
                f"Section edge drift {float(details['worst_section_edge_distance_px']):.2f}px / {float(details['max_section_edge_distance_px']):.2f}px"
            )
        elif (
            bool(details.get("edge_distance_gate_active", False))
            and _metric_exceeds_limit(details, "mean_edge_distance_px", "max_mean_edge_distance_px")
            and details.get("mean_edge_distance_px") is not None
            and details.get("max_mean_edge_distance_px") is not None
        ):
            lines.append(
                f"Edge drift {float(details['mean_edge_distance_px']):.2f}px / {float(details['max_mean_edge_distance_px']):.2f}px"
            )
        elif bool(details.get("ssim_gate_active", False)) and _metric_exceeds_limit(details, "ssim", "min_ssim", high_is_bad=False) and details.get("ssim") is not None and details.get("min_ssim") is not None:
            lines.append(f"Similarity {float(details['ssim']):.3f} / {float(details['min_ssim']):.3f}")
        elif bool(details.get("mse_gate_active", False)) and _metric_exceeds_limit(details, "mse", "max_mse") and details.get("mse") is not None and details.get("max_mse") is not None:
            lines.append(f"Difference {float(details['mse']):.1f} / {float(details['max_mse']):.1f}")
        elif bool(details.get("anomaly_gate_active", False)) and _metric_exceeds_limit(details, "anomaly_score", "min_anomaly_score", high_is_bad=False) and details.get("anomaly_score") is not None and details.get("min_anomaly_score") is not None:
            lines.append(
                f"Anomaly {float(details['anomaly_score']):.3f} / {float(details['min_anomaly_score']):.3f}"
            )

    if len(lines) < 3:
        lines.append(f"Coverage {required_coverage:.1%} / {min_required_coverage:.1%}")
    if len(lines) < 3:
        lines.append(f"Outside print {outside_allowed_ratio:.1%} / {max_outside_allowed_ratio:.1%}")
    return lines[:3]


def determine_operator_outcome(passed: bool, details: dict) -> ProductionOutcome:
    if passed:
        return ProductionOutcome(GOOD, "GOOD", None, _make_summary_lines(GOOD, None, details))

    if str(details.get("failure_stage", "")).lower() == "registration" or _registration_rejection_reason(details):
        return ProductionOutcome(
            REVIEW,
            "CHECK PLACEMENT",
            REASON_REGISTRATION_FAILURE,
            _make_summary_lines(REVIEW, REASON_REGISTRATION_FAILURE, details),
        )

    failures = _failed_checks(details)
    primary_reason = failures[0] if failures else REASON_REFERENCE_MISMATCH

    if failures and len(failures) == 1 and _is_borderline_failure(details, primary_reason):
        return ProductionOutcome(
            REVIEW,
            "NEEDS REVIEW - RED BIN",
            primary_reason,
            _make_summary_lines(REVIEW, primary_reason, details),
        )

    return ProductionOutcome(
        REJECT,
        "REJECT",
        primary_reason,
        _make_summary_lines(REJECT, primary_reason, details),
    )
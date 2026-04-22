#!/usr/bin/env python3

from inspection_system.app.registration_schema import (
    get_registration_commissioning_config,
    get_registration_config,
)


def _required_anchor_count(anchor_mode: str) -> int:
    return {
        "none": 0,
        "single": 1,
        "pair": 2,
        "multi": 3,
    }.get(str(anchor_mode).strip().lower(), 0)


def _anchor_has_search_window(anchor: dict) -> bool:
    search_window = anchor.get("search_window", {}) if isinstance(anchor, dict) else {}
    try:
        return int(search_window.get("width", 0)) > 0 and int(search_window.get("height", 0)) > 0
    except (TypeError, ValueError):
        return False


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def registration_baseline_matches_config(stored_baseline: dict | None, current_summary: dict) -> bool:
    if not isinstance(stored_baseline, dict):
        return False

    comparable_keys = [
        "runtime_mode",
        "strategy",
        "transform_model",
        "anchor_mode",
        "required_anchor_count",
        "enabled_anchor_count",
        "anchor_ids",
        "search_margin_px",
        "datum_origin",
        "datum_orientation",
        "requires_datum_confirmation",
        "datum_confirmed",
        "requires_expected_transform_validation",
        "expected_transform_confirmed",
    ]
    for key in comparable_keys:
        if stored_baseline.get(key) != current_summary.get(key):
            return False

    stored_expected = stored_baseline.get("expected_transform", {})
    current_expected = current_summary.get("expected_transform", {})
    if not isinstance(stored_expected, dict) or not isinstance(current_expected, dict):
        return False

    for key in ("max_angle_deg", "max_shift_x", "max_shift_y"):
        if stored_expected.get(key) != current_expected.get(key):
            return False
    return True


def build_registration_commissioning_summary(config: dict) -> dict:
    alignment_cfg = config.get("alignment", {}) if isinstance(config, dict) else {}
    registration_cfg = get_registration_config(config)
    commissioning_cfg = get_registration_commissioning_config(config)
    strategy = str(registration_cfg.get("strategy", "moments")).lower()
    runtime_mode = str(alignment_cfg.get("mode", strategy)).lower()
    anchor_mode = str(registration_cfg.get("anchor_mode", "none")).lower()
    enabled_anchors = [
        anchor
        for anchor in registration_cfg.get("anchors", [])
        if isinstance(anchor, dict) and bool(anchor.get("enabled", True))
    ]
    required_anchor_count = _required_anchor_count(anchor_mode)
    active_quality_gates = [
        key
        for key, value in registration_cfg.get("quality_gates", {}).items()
        if value not in {None, ""}
    ]
    datum_origin = str(registration_cfg["datum_frame"]["origin"]).lower()
    datum_orientation = str(registration_cfg["datum_frame"]["orientation"]).lower()
    expected_transform = {
        "max_angle_deg": _safe_float(alignment_cfg.get("max_angle_deg", 1.0), 1.0),
        "max_shift_x": _safe_int(alignment_cfg.get("max_shift_x", 4), 4),
        "max_shift_y": _safe_int(alignment_cfg.get("max_shift_y", 3), 3),
    }
    requires_datum_confirmation = required_anchor_count > 0 or datum_origin != "roi_top_left" or datum_orientation != "part_axis"
    requires_expected_transform_validation = runtime_mode != "moments" or strategy != "moments" or required_anchor_count > 0
    datum_confirmed = bool(commissioning_cfg["datum_confirmed"])
    expected_transform_confirmed = bool(commissioning_cfg["expected_transform_confirmed"])
    expected_transform_limits_ready = (
        expected_transform["max_angle_deg"] >= 0.0
        and expected_transform["max_shift_x"] >= 0
        and expected_transform["max_shift_y"] >= 0
    )

    issues: list[str] = []
    actions: list[str] = []

    if strategy in {"anchor_translation", "anchor_pair"} and required_anchor_count <= 0:
        issues.append(f"registration strategy '{strategy}' requires registration anchors")
        actions.append("Set Registration Anchor Mode and define anchor locations before relying on production results.")

    if required_anchor_count > 0 and len(enabled_anchors) < required_anchor_count:
        issues.append(
            f"enabled anchors {len(enabled_anchors)}/{required_anchor_count} for {anchor_mode} registration"
        )
        actions.append(
            f"Define at least {required_anchor_count} enabled registration anchors before relying on {strategy}."
        )

    missing_search_windows = [
        str(anchor.get("anchor_id", "anchor"))
        for anchor in enabled_anchors[:required_anchor_count]
        if not _anchor_has_search_window(anchor)
    ]
    if missing_search_windows:
        issues.append(f"search windows missing for {', '.join(missing_search_windows)}")
        actions.append("Define search windows for the enabled registration anchors.")

    if requires_datum_confirmation and not datum_confirmed:
        issues.append("datum frame not confirmed")
        actions.append("Confirm the datum frame in Registration Setup after reviewing the anchor-origin/orientation mapping.")

    if requires_expected_transform_validation:
        if not expected_transform_limits_ready:
            issues.append("expected transform limits invalid")
            actions.append("Set non-negative expected angle and shift limits before relying on production results.")
        elif not expected_transform_confirmed:
            issues.append("expected transform limits not validated")
            actions.append("Review the expected angle/shift limits and confirm them in Registration Setup.")

    checklist = [
        {
            "key": "anchors",
            "label": "Anchor placement",
            "required": required_anchor_count > 0,
            "ready": required_anchor_count == 0 or len(enabled_anchors) >= required_anchor_count,
            "summary": (
                "not required"
                if required_anchor_count == 0
                else f"{len(enabled_anchors)}/{required_anchor_count} enabled anchors"
            ),
        },
        {
            "key": "search_windows",
            "label": "Search windows",
            "required": required_anchor_count > 0,
            "ready": required_anchor_count == 0 or not missing_search_windows,
            "summary": (
                "not required"
                if required_anchor_count == 0
                else "all required anchor windows defined"
                if not missing_search_windows
                else f"missing for {', '.join(missing_search_windows)}"
            ),
        },
        {
            "key": "datum",
            "label": "Datum confirmation",
            "required": requires_datum_confirmation,
            "ready": (not requires_datum_confirmation) or datum_confirmed,
            "summary": (
                "default datum"
                if not requires_datum_confirmation
                else "confirmed"
                if datum_confirmed
                else "pending confirmation"
            ),
        },
        {
            "key": "expected_transform",
            "label": "Expected transform validation",
            "required": requires_expected_transform_validation,
            "ready": (not requires_expected_transform_validation) or (expected_transform_limits_ready and expected_transform_confirmed),
            "summary": (
                "default moments tolerance"
                if not requires_expected_transform_validation
                else "validated"
                if expected_transform_limits_ready and expected_transform_confirmed
                else "pending validation"
            ),
        },
    ]

    summary_parts = [
        f"runtime {runtime_mode}",
        f"requested {strategy}",
    ]
    if required_anchor_count > 0:
        summary_parts.append(f"anchors {len(enabled_anchors)}/{required_anchor_count}")
    else:
        summary_parts.append("anchors off")
    summary_parts.append(f"datum {datum_origin}/{datum_orientation}")
    if requires_datum_confirmation:
        summary_parts.append("datum ok" if datum_confirmed else "datum pending")
    if requires_expected_transform_validation:
        summary_parts.append("transform ok" if expected_transform_confirmed else "transform pending")
    if active_quality_gates:
        summary_parts.append("quality gates on")

    return {
        "runtime_mode": runtime_mode,
        "strategy": strategy,
        "transform_model": str(registration_cfg.get("transform_model", "rigid")).lower(),
        "anchor_mode": anchor_mode,
        "required_anchor_count": required_anchor_count,
        "enabled_anchor_count": len(enabled_anchors),
        "anchor_ids": [str(anchor.get("anchor_id", "anchor")) for anchor in enabled_anchors],
        "search_margin_px": int(registration_cfg.get("search_margin_px", 0)),
        "datum_origin": datum_origin,
        "datum_orientation": datum_orientation,
        "requires_datum_confirmation": requires_datum_confirmation,
        "datum_confirmed": datum_confirmed,
        "requires_expected_transform_validation": requires_expected_transform_validation,
        "expected_transform_confirmed": expected_transform_confirmed,
        "expected_transform": expected_transform,
        "quality_gates": dict(registration_cfg.get("quality_gates", {})),
        "checklist": checklist,
        "ready": not issues,
        "issues": issues,
        "actions": actions,
        "summary": " | ".join(summary_parts),
    }
#!/usr/bin/env python3
"""Shared registration schema helpers for registration-first commissioning."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_REGISTRATION_CONFIG = {
    "strategy": "moments",
    "transform_model": "rigid",
    "anchor_mode": "none",
    "subpixel_refinement": "off",
    "search_margin_px": 24,
    "anchors": [],
    "quality_gates": {
        "min_confidence": None,
        "max_mean_residual_px": None,
    },
    "datum_frame": {
        "origin": "roi_top_left",
        "orientation": "part_axis",
    },
}

DEFAULT_REGISTRATION_COMMISSIONING = {
    "datum_confirmed": False,
    "expected_transform_confirmed": False,
}


def _deep_merge_dicts(defaults: dict, overrides: dict) -> dict:
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_int(value, default: int, *, allow_none: bool = False):
    if value is None and allow_none:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None if allow_none else int(default)


def _coerce_float(value, default: float | None, *, allow_none: bool = False):
    if value is None and allow_none:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None if allow_none else default


def _normalize_anchor(anchor: dict, index: int) -> dict:
    reference_point = anchor.get("reference_point", {}) if isinstance(anchor.get("reference_point"), dict) else {}
    search_window = anchor.get("search_window", {}) if isinstance(anchor.get("search_window"), dict) else {}
    anchor_id = str(anchor.get("anchor_id") or anchor.get("id") or f"anchor_{index + 1}").strip() or f"anchor_{index + 1}"
    label = str(anchor.get("label") or anchor_id.replace("_", " ").title()).strip() or anchor_id
    return {
        "anchor_id": anchor_id,
        "label": label,
        "kind": str(anchor.get("kind", "feature")).lower(),
        "enabled": bool(anchor.get("enabled", True)),
        "reference_point": {
            "x": _coerce_int(reference_point.get("x", anchor.get("x", 0)), 0),
            "y": _coerce_int(reference_point.get("y", anchor.get("y", 0)), 0),
        },
        "search_window": {
            "x": _coerce_int(search_window.get("x", 0), 0),
            "y": _coerce_int(search_window.get("y", 0), 0),
            "width": _coerce_int(search_window.get("width", 0), 0),
            "height": _coerce_int(search_window.get("height", 0), 0),
        },
    }


def normalize_registration_anchors(raw_anchors) -> list[dict]:
    if not isinstance(raw_anchors, list):
        raw_anchors = []
    return [
        _normalize_anchor(anchor, index)
        for index, anchor in enumerate(raw_anchors)
        if isinstance(anchor, dict)
    ]


def get_registration_commissioning_config(config: dict) -> dict:
    alignment_cfg = config.get("alignment", {}) if isinstance(config, dict) else {}
    raw_registration = alignment_cfg.get("registration", {})
    if not isinstance(raw_registration, dict):
        raw_registration = {}
    raw_commissioning = raw_registration.get("commissioning", {})
    if not isinstance(raw_commissioning, dict):
        raw_commissioning = {}
    return {
        "datum_confirmed": bool(
            raw_commissioning.get(
                "datum_confirmed",
                DEFAULT_REGISTRATION_COMMISSIONING["datum_confirmed"],
            )
        ),
        "expected_transform_confirmed": bool(
            raw_commissioning.get(
                "expected_transform_confirmed",
                DEFAULT_REGISTRATION_COMMISSIONING["expected_transform_confirmed"],
            )
        ),
    }


def get_registration_config(config: dict) -> dict:
    alignment_cfg = config.get("alignment", {}) if isinstance(config, dict) else {}
    raw_registration = alignment_cfg.get("registration", {})
    if not isinstance(raw_registration, dict):
        raw_registration = {}

    registration_cfg = _deep_merge_dicts(DEFAULT_REGISTRATION_CONFIG, raw_registration)
    registration_cfg["strategy"] = str(
        raw_registration.get("strategy", alignment_cfg.get("mode", DEFAULT_REGISTRATION_CONFIG["strategy"]))
    ).lower()
    registration_cfg["transform_model"] = str(
        registration_cfg.get("transform_model", DEFAULT_REGISTRATION_CONFIG["transform_model"])
    ).lower()
    registration_cfg["anchor_mode"] = str(
        registration_cfg.get("anchor_mode", DEFAULT_REGISTRATION_CONFIG["anchor_mode"])
    ).lower()
    registration_cfg["subpixel_refinement"] = str(
        registration_cfg.get("subpixel_refinement", DEFAULT_REGISTRATION_CONFIG["subpixel_refinement"])
    ).lower()
    registration_cfg["search_margin_px"] = _coerce_int(
        registration_cfg.get("search_margin_px", DEFAULT_REGISTRATION_CONFIG["search_margin_px"]),
        DEFAULT_REGISTRATION_CONFIG["search_margin_px"],
    )

    quality_gates = registration_cfg.get("quality_gates", {})
    if not isinstance(quality_gates, dict):
        quality_gates = {}
    registration_cfg["quality_gates"] = {
        "min_confidence": _coerce_float(quality_gates.get("min_confidence"), None, allow_none=True),
        "max_mean_residual_px": _coerce_float(quality_gates.get("max_mean_residual_px"), None, allow_none=True),
    }

    datum_frame = registration_cfg.get("datum_frame", {})
    if not isinstance(datum_frame, dict):
        datum_frame = {}
    registration_cfg["datum_frame"] = {
        "origin": str(datum_frame.get("origin", DEFAULT_REGISTRATION_CONFIG["datum_frame"]["origin"])).lower(),
        "orientation": str(
            datum_frame.get("orientation", DEFAULT_REGISTRATION_CONFIG["datum_frame"]["orientation"])
        ).lower(),
    }

    raw_anchors = raw_registration.get("anchors", registration_cfg.get("anchors", []))
    registration_cfg["anchors"] = normalize_registration_anchors(raw_anchors)
    registration_cfg.pop("commissioning", None)
    return registration_cfg


def build_alignment_metadata(config: dict) -> dict:
    alignment_cfg = config.get("alignment", {}) if isinstance(config, dict) else {}
    return {
        "enabled": bool(alignment_cfg.get("enabled", True)),
        "mode": str(alignment_cfg.get("mode", "moments")).lower(),
        "tolerance_profile": str(alignment_cfg.get("tolerance_profile", "balanced")).lower(),
        "max_angle_deg": _coerce_float(alignment_cfg.get("max_angle_deg", 1.0), 1.0),
        "max_shift_x": _coerce_int(alignment_cfg.get("max_shift_x", 4), 4),
        "max_shift_y": _coerce_int(alignment_cfg.get("max_shift_y", 3), 3),
        "registration": get_registration_config(config),
    }
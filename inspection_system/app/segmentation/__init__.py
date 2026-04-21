from __future__ import annotations

from inspection_system.app.segmentation.binary_threshold import apply_binary_threshold
from inspection_system.app.segmentation.binary_threshold_inverted import apply_binary_threshold_inverted


SEGMENTATION_STRATEGIES = {
    "binary_threshold": apply_binary_threshold,
    "binary_threshold_inverted": apply_binary_threshold_inverted,
}

LEGACY_THRESHOLD_MODES = {
    "fixed": ("binary_threshold", "fixed"),
    "otsu": ("binary_threshold", "otsu"),
    "fixed_inv": ("binary_threshold_inverted", "fixed"),
    "otsu_inv": ("binary_threshold_inverted", "otsu"),
}


def resolve_segmentation_settings(inspection_cfg: dict) -> dict:
    strategy_name = str(inspection_cfg.get("segmentation_strategy", "")).strip().lower()
    threshold_mode = str(inspection_cfg.get("threshold_mode", "fixed")).strip().lower()

    legacy_strategy_name, legacy_threshold_method = LEGACY_THRESHOLD_MODES.get(
        threshold_mode,
        ("binary_threshold", "fixed"),
    )

    if strategy_name not in SEGMENTATION_STRATEGIES:
        strategy_name = legacy_strategy_name

    threshold_method = str(inspection_cfg.get("threshold_method", "")).strip().lower()
    if threshold_method not in {"fixed", "otsu"}:
        threshold_method = legacy_threshold_method

    threshold_value = int(inspection_cfg.get("threshold_value", 180))
    return {
        "strategy_name": strategy_name,
        "threshold_method": threshold_method,
        "threshold_value": threshold_value,
    }


def build_legacy_threshold_mode(settings: dict) -> str:
    strategy_name = str(settings.get("strategy_name", "binary_threshold")).strip().lower()
    threshold_method = str(settings.get("threshold_method", "fixed")).strip().lower()

    if strategy_name == "binary_threshold_inverted":
        return "otsu_inv" if threshold_method == "otsu" else "fixed_inv"
    return "otsu" if threshold_method == "otsu" else "fixed"


def apply_segmentation_strategy(gray, inspection_cfg: dict, cv2):
    settings = resolve_segmentation_settings(inspection_cfg)
    strategy = SEGMENTATION_STRATEGIES[settings["strategy_name"]]
    mask = strategy(
        gray,
        threshold_method=settings["threshold_method"],
        threshold_value=settings["threshold_value"],
        cv2=cv2,
    )
    return mask, settings


__all__ = [
    "apply_segmentation_strategy",
    "build_legacy_threshold_mode",
    "resolve_segmentation_settings",
]
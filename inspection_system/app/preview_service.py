#!/usr/bin/env python3
"""Shared preview-image discovery helpers for UI surfaces."""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image, ImageStat
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageStat = None
    PIL_AVAILABLE = False


PREVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
REFERENCE_PREVIEW_NAME = "golden_reference_image.png"


def is_informative_preview_image(image_path: Path) -> bool:
    """Return False for likely blank/black preview images."""
    if not PIL_AVAILABLE:
        return True
    try:
        image = Image.open(image_path).convert("L")
        min_px, max_px = image.getextrema()
        mean_px = float(ImageStat.Stat(image).mean[0])
    except Exception:
        return False

    contrast = int(max_px) - int(min_px)
    return contrast >= 6 and mean_px >= 6.0


def find_preview_image(reference_dir: Path, is_informative_fn=None) -> Path | None:
    if not reference_dir.exists():
        return None

    is_informative = is_informative_fn or is_informative_preview_image

    preferred = reference_dir / REFERENCE_PREVIEW_NAME
    if preferred.exists() and is_informative(preferred):
        return preferred

    candidates = [
        path for path in reference_dir.iterdir()
        if path.is_file() and path.suffix.lower() in PREVIEW_EXTENSIONS
    ]
    if not candidates:
        return None

    # Avoid showing debug diff/mask snapshots when a real sample image is available.
    non_debug = [
        path for path in candidates
        if not path.name.endswith("_diff.png") and not path.name.endswith("_mask.png")
    ]

    informative_non_debug = [path for path in non_debug if is_informative(path)]
    if informative_non_debug:
        return max(informative_non_debug, key=lambda path: path.stat().st_mtime)

    informative_candidates = [path for path in candidates if is_informative(path)]
    if informative_candidates:
        return max(informative_candidates, key=lambda path: path.stat().st_mtime)

    if non_debug:
        return max(non_debug, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def describe_preview_image(preview_path: Path) -> str:
    name = preview_path.name
    if name == REFERENCE_PREVIEW_NAME:
        return "reference"
    if name.endswith("_diff.png"):
        return "difference debug"
    if name.endswith("_mask.png"):
        return "mask debug"
    return "latest sample"
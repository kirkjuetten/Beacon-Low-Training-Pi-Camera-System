#!/usr/bin/env python3
"""Legacy frame-acquisition shim.

Historically this module owned the ``rpicam-still`` subprocess flow and a
polled file lock. As of Phase 2 (camera backend hardening) the actual logic
lives in :mod:`inspection_system.app.camera.backend`. This module remains as
a thin compatibility layer so the eight existing call sites continue to work
unchanged while inheriting the new resilience improvements:

* JSON lock files with ``pid`` and timestamp.
* Automatic reaping of stale locks (older than ``2 * max_wait_s``).
* :mod:`atexit` cleanup of locks owned by the current process.
* Wall-clock latency measurement (returned via :func:`capture_frame`).

New code should prefer :func:`capture_frame`, which returns a structured
:class:`~inspection_system.app.camera.backend.CaptureResult`.
"""
from __future__ import annotations

from pathlib import Path

from inspection_system.app.camera.backend import (
    CaptureResult,
    RpicamStillBackend,
)
from inspection_system.app.camera_interface import TEMP_IMAGE

__all__ = [
    "capture_to_temp",
    "capture_frame",
    "cleanup_temp_image",
    "TEMP_IMAGE",
]


# A single shared backend keeps the atexit lock cleanup registered exactly
# once for the lifetime of the process.
_default_backend = RpicamStillBackend(temp_image=TEMP_IMAGE)


def cleanup_temp_image() -> None:
    """Remove the active temp capture image, if present."""
    if TEMP_IMAGE.exists():
        try:
            TEMP_IMAGE.unlink()
        except OSError:
            pass


def capture_frame(config: dict) -> CaptureResult:
    """Capture a single frame using the default backend.

    Preferred over :func:`capture_to_temp` for new code: returns a
    :class:`CaptureResult` with structured error_kind and latency_ms.
    """
    return _default_backend.capture(config)


def capture_to_temp(config: dict) -> tuple[int, Path, str]:
    """Backward-compatible wrapper around :func:`capture_frame`.

    Returns the historical ``(return_code, image_path, stderr)`` tuple so
    existing call sites do not need to change in this phase.
    """
    result = _default_backend.capture(config)
    return result.return_code, result.image_path, result.stderr


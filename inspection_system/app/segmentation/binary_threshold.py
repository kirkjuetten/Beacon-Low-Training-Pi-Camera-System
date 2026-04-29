"""Back-compat shim: canonical implementation lives in ``thresholding``."""
from __future__ import annotations

from inspection_system.app.segmentation.thresholding import apply_binary_threshold

__all__ = ["apply_binary_threshold"]

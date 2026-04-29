"""Per-session training event logger.

Extracted from :mod:`inspection_system.app.interactive_training` in Phase 4
to give the logger a clean, testable home. The original location continues
to re-export :class:`TrainingLogger` so existing call sites and tests do not
need to change.

The logger writes plain text lines to a per-session file rooted under
``log_dir``. It is intentionally simple -- the inspection pipeline produces
human-readable lines that operators can grep and tail, with optional
metric columns appended via
:func:`inspection_system.app.metrics_format.append_optional_metric` so that
``None`` values do not crash format strings at runtime.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

from inspection_system.app.metrics_format import append_optional_metric


class TrainingLogger:
    """Logs training sessions and decisions for analysis."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.current_session = None
        self.session_start_time = None

    def start_session(self):
        """Start a new training session."""
        self.session_start_time = time.time()
        session_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.session_start_time))
        self.current_session = f"training_session_{session_id}.log"
        self._log(f"=== TRAINING SESSION STARTED: {session_id} ===")

    @staticmethod
    def _append_optional_metric(log_entry: str, details: dict, key: str, template: str) -> str:
        """Back-compat wrapper around :func:`append_optional_metric`.

        Kept on the class so external callers that monkeypatched this
        staticmethod (or that vendored TrainingLogger before Phase 4) still
        work. New code should call the module-level helper directly.
        """
        return append_optional_metric(log_entry, details, key, template)

    def log_inspection(self, image_path: Path, passed: bool, details: dict, feedback: str, description: str):
        """Log an inspection result and feedback."""
        if not self.current_session:
            self.start_session()

        timestamp = time.strftime("%H:%M:%S", time.localtime(time.time()))
        status = "PASS" if passed else "FAIL"

        log_entry = f"[{timestamp}] {status} -> {feedback.upper()}"
        log_entry += f" | {Path(image_path).name}"
        log_entry += f" | Coverage: {details.get('required_coverage', 0):.3f}"
        log_entry += f" | Outside: {details.get('outside_allowed_ratio', 0):.3f}"
        log_entry = append_optional_metric(log_entry, details, 'mean_edge_distance_px', " | EdgeDist: {value:.3f}px")
        log_entry = append_optional_metric(log_entry, details, 'worst_section_edge_distance_px', " | SectEdge: {value:.3f}px")
        log_entry = append_optional_metric(log_entry, details, 'worst_section_width_delta_ratio', " | SectWidth: {value:.1%}")
        log_entry = append_optional_metric(log_entry, details, 'worst_section_center_offset_px', " | SectCenter: {value:.3f}px")
        log_entry = append_optional_metric(log_entry, details, 'ssim', " | SSIM: {value:.3f}")
        log_entry = append_optional_metric(log_entry, details, 'anomaly_score', " | Anomaly: {value:.3f}")

        log_entry += f" | {description}"

        self._log(log_entry)

    def log_threshold_suggestion(self, suggestions: dict):
        """Log threshold adjustment suggestions."""
        if suggestions:
            self._log("=== THRESHOLD SUGGESTIONS ===")
            for key, value in suggestions.items():
                self._log(f"Suggested {key}: {value:.4f}")
            self._log("=" * 30)

    def log_review_findings(self, warnings: list[str]):
        """Log review-stage warnings about config fit and prerequisites."""
        if warnings:
            self._log("=== REVIEW WARNINGS ===")
            for warning in warnings:
                self._log(warning)
            self._log("=" * 30)

    def end_session(self):
        """End the current training session."""
        if self.current_session:
            duration = time.time() - self.session_start_time
            self._log(f"=== SESSION ENDED: Duration {duration:.1f}s ===")
            self.current_session = None

    def _log(self, message: str):
        """Write message to current session log."""
        if self.current_session:
            log_path = self.log_dir / self.current_session
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    def get_session_summary(self) -> Dict[str, int]:
        """Get summary of current session decisions."""
        if not self.current_session:
            return {}

        log_path = self.log_dir / self.current_session
        if not log_path.exists():
            return {}

        summary = {'approve': 0, 'reject': 0, 'review': 0, 'total': 0}

        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '-> APPROVE' in line:
                    summary['approve'] += 1
                elif '-> REJECT' in line:
                    summary['reject'] += 1
                elif '-> REVIEW' in line:
                    summary['review'] += 1
                summary['total'] += 1 if any(x in line for x in ['PASS', 'FAIL']) else 0

        return summary

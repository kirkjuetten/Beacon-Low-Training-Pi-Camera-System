"""End-of-session summary helpers for production runs.

Purpose
-------
Operator-facing summary printed when a production session ends. Surfaces the
counters the floor cares about (good/reject/review/total), the inspection
latency distribution (count, mean, p50, p95), and how many indicator I/O
calls failed during the session. Pure data + formatting; no pygame, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class LatencyTracker:
    """Records per-inspection latency samples in milliseconds.

    Designed for short production sessions (hundreds to low thousands of
    inspections), so we keep all samples and compute percentiles on demand.
    """

    samples_ms: list[float] = field(default_factory=list)

    def record(self, latency_ms: Optional[float]) -> None:
        if latency_ms is None:
            return
        try:
            value = float(latency_ms)
        except (TypeError, ValueError):
            return
        if value < 0:
            return
        self.samples_ms.append(value)

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    @property
    def mean_ms(self) -> Optional[float]:
        if not self.samples_ms:
            return None
        return sum(self.samples_ms) / len(self.samples_ms)

    def percentile(self, p: float) -> Optional[float]:
        """Return the requested percentile (0-100) using nearest-rank.

        Nearest-rank is operator-friendly for small N: the returned value is
        always an actual observed sample, so it is easy to relate back to a
        specific inspection on the line.
        """
        if not self.samples_ms:
            return None
        if p < 0:
            p = 0.0
        if p > 100:
            p = 100.0
        ordered = sorted(self.samples_ms)
        if p == 0:
            return ordered[0]
        # Nearest-rank: ceil(p/100 * N), 1-indexed.
        n = len(ordered)
        rank = max(1, min(n, int(-(-p * n // 100))))
        return ordered[rank - 1]

    @property
    def p50_ms(self) -> Optional[float]:
        return self.percentile(50)

    @property
    def p95_ms(self) -> Optional[float]:
        return self.percentile(95)


def _format_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f} ms"


def format_run_summary(
    totals,
    latency: LatencyTracker,
    indicator_errors: int = 0,
    *,
    title: str = "Production session summary",
) -> str:
    """Render an operator-readable summary block.

    ``totals`` is duck-typed against ``production_screen.CounterScope``:
    we read ``total``, ``good``, ``reject``, ``review``. The indicator error
    count is the number of I/O calls (``pulse_pass`` / ``pulse_fail``) that
    raised during the session; non-zero values point to a wiring or RS-485
    bus issue that the operator should report before the next shift.
    """
    total = int(getattr(totals, "total", 0) or 0)
    good = int(getattr(totals, "good", 0) or 0)
    reject = int(getattr(totals, "reject", 0) or 0)
    review = int(getattr(totals, "review", 0) or 0)

    if total > 0:
        pass_rate = 100.0 * good / total
        rate_text = f"{pass_rate:.1f}%"
    else:
        rate_text = "n/a"

    if total > 0:
        error_rate = 100.0 * indicator_errors / total
        err_text = f"{indicator_errors} ({error_rate:.1f}% of inspections)"
    else:
        err_text = f"{indicator_errors}"

    lines: list[str] = [
        title,
        "-" * len(title),
        f"Inspections : {total}",
        f"  Good      : {good}",
        f"  Reject    : {reject}",
        f"  Review    : {review}",
        f"  Pass rate : {rate_text}",
        "Latency",
        f"  Samples   : {latency.count}",
        f"  Mean      : {_format_ms(latency.mean_ms)}",
        f"  p50       : {_format_ms(latency.p50_ms)}",
        f"  p95       : {_format_ms(latency.p95_ms)}",
        f"Indicator I/O errors : {err_text}",
    ]
    return "\n".join(lines)


def iter_recorded_samples(latency: LatencyTracker) -> Iterable[float]:
    """Convenience accessor used by tests; intentionally not a property."""
    return tuple(latency.samples_ms)

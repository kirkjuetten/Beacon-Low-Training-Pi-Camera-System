"""Tests for inspection_system.app.run_summary."""
from __future__ import annotations

from dataclasses import dataclass

from inspection_system.app.run_summary import (
    LatencyTracker,
    format_run_summary,
)


@dataclass
class _Totals:
    total: int = 0
    good: int = 0
    reject: int = 0
    review: int = 0


def test_latency_tracker_empty() -> None:
    lt = LatencyTracker()
    assert lt.count == 0
    assert lt.mean_ms is None
    assert lt.p50_ms is None
    assert lt.p95_ms is None
    assert lt.percentile(50) is None


def test_latency_tracker_records_and_ignores_invalid() -> None:
    lt = LatencyTracker()
    lt.record(10)
    lt.record(20.5)
    lt.record(None)
    lt.record(-1)  # negative ignored
    lt.record("not-a-number")  # type: ignore[arg-type]
    assert lt.count == 2
    assert lt.mean_ms is not None
    assert abs(lt.mean_ms - 15.25) < 1e-9


def test_latency_tracker_single_sample() -> None:
    lt = LatencyTracker()
    lt.record(42.0)
    assert lt.p50_ms == 42.0
    assert lt.p95_ms == 42.0
    assert lt.mean_ms == 42.0


def test_latency_tracker_percentiles_nearest_rank() -> None:
    lt = LatencyTracker()
    for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        lt.record(v)
    # nearest-rank: p50 -> ceil(0.5*10)=5 -> sorted[4]=50
    assert lt.p50_ms == 50.0
    # p95 -> ceil(0.95*10)=10 -> sorted[9]=100
    assert lt.p95_ms == 100.0
    # p10 -> ceil(0.1*10)=1 -> sorted[0]=10
    assert lt.percentile(10) == 10.0


def test_latency_tracker_percentile_clamps_bounds() -> None:
    lt = LatencyTracker()
    for v in [5, 15, 25]:
        lt.record(v)
    assert lt.percentile(-10) == 5.0
    assert lt.percentile(0) == 5.0
    assert lt.percentile(150) == 25.0
    assert lt.percentile(100) == 25.0


def test_format_run_summary_with_inspections() -> None:
    totals = _Totals(total=10, good=8, reject=1, review=1)
    lt = LatencyTracker()
    for v in [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]:
        lt.record(v)
    out = format_run_summary(totals, lt, indicator_errors=0)
    assert "Inspections : 10" in out
    assert "Good      : 8" in out
    assert "Reject    : 1" in out
    assert "Review    : 1" in out
    assert "Pass rate : 80.0%" in out
    assert "Samples   : 10" in out
    assert "Mean      : 95 ms" in out
    assert "p50       : 90 ms" in out
    assert "p95       : 140 ms" in out
    assert "Indicator I/O errors : 0" in out


def test_format_run_summary_handles_zero_inspections() -> None:
    totals = _Totals()
    lt = LatencyTracker()
    out = format_run_summary(totals, lt, indicator_errors=0)
    assert "Inspections : 0" in out
    assert "Pass rate : n/a" in out
    assert "Mean      : n/a" in out
    assert "p50       : n/a" in out
    assert "p95       : n/a" in out


def test_format_run_summary_indicator_errors_show_rate() -> None:
    totals = _Totals(total=20, good=18, reject=2)
    lt = LatencyTracker()
    for _ in range(20):
        lt.record(50.0)
    out = format_run_summary(totals, lt, indicator_errors=3)
    assert "Indicator I/O errors : 3 (15.0% of inspections)" in out


def test_format_run_summary_custom_title() -> None:
    out = format_run_summary(_Totals(), LatencyTracker(), 0, title="Shift recap")
    assert out.startswith("Shift recap\n-----------\n")


def test_format_run_summary_tolerates_duck_typed_totals() -> None:
    class Bag:
        total = 3
        good = 2
        reject = 1
        review = 0

    out = format_run_summary(Bag(), LatencyTracker(), 0)
    assert "Inspections : 3" in out
    assert "Pass rate : 66.7%" in out


def test_format_run_summary_handles_missing_attributes() -> None:
    class Empty:
        pass

    out = format_run_summary(Empty(), LatencyTracker(), 0)
    assert "Inspections : 0" in out
    assert "Pass rate : n/a" in out

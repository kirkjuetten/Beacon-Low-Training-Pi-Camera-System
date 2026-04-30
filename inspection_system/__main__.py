"""``python -m inspection_system`` entrypoint.

Currently exposes a single subcommand:

    python -m inspection_system --self-test

Which runs an end-to-end smoke test that exercises every subsystem touched
during a real inspection cycle, but without requiring any hardware. It is
the canonical "is this Pi ready for a recipe?" preflight, and the same
script CI runs as a smoke gate.

What the self-test verifies
---------------------------
1. Config loads successfully from disk.
2. The IndicatorBus dispatches and pulses (uses the safe ``none`` mode so
   the test does not need the Waveshare RS-485 converter to be present).
3. The camera abstraction round-trips through :class:`MockCameraBackend`
   so the :class:`CaptureResult` plumbing is exercised end to end.
4. Pilot-readiness can be evaluated. The result is reported but does *not*
   fail the self-test, because a fresh checkout legitimately has no
   commissioned recipe yet.

Exit codes: 0 = all checks passed, 1 = one or more failed.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable


def _check(label: str, fn: Callable[[], object]) -> tuple[str, bool, str]:
    """Run a check and return (label, ok, detail). Never raises."""
    try:
        detail = fn()
    except Exception as exc:  # pragma: no cover - hit when checks crash
        return label, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return label, True, str(detail) if detail is not None else "ok"


def _check_config_loads() -> str:
    from inspection_system.app.camera_interface import load_config

    config = load_config()
    if not isinstance(config, dict):
        raise RuntimeError("load_config did not return a dict")
    return f"loaded {len(config)} top-level keys"


def _check_indicator_bus() -> str:
    from inspection_system.app.io.indicator_bus import build_indicator_bus

    # Force the safe mode for the self-test so we never touch real serial
    # ports or GPIO pins.
    bus = build_indicator_bus({"io": {"mode": "none"}})
    bus.pulse_pass()
    bus.pulse_fail()
    bus.cleanup()
    return f"bus={type(bus).__name__} enabled={getattr(bus, 'enabled', '?')}"


def _check_mock_camera_backend() -> str:
    from inspection_system.app.camera.backend import MockCameraBackend

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "self_test.jpg"
        backend = MockCameraBackend(temp_image=target)
        result = backend.capture({})
        if not result.ok:
            raise RuntimeError(f"mock capture returned non-zero: {result}")
        if not target.exists():
            raise RuntimeError("mock capture did not write its temp image")
        backend.cleanup()
    return f"latency_ms={result.latency_ms:.2f} error_kind={result.error_kind}"


def _check_pilot_readiness() -> str:
    from inspection_system.app.camera_interface import load_config
    from inspection_system.app.pilot_readiness import build_supervised_pilot_status

    config = load_config()
    status = build_supervised_pilot_status(config)
    state = "READY" if status.get("ready") else "NOT READY (informational)"
    return f"{state} project={status.get('current_project') or '<none>'}"


def _wait_for_trigger(seconds: float) -> int:
    """Live trigger probe: build the trigger from camera_config.json and watch.

    This is the operator's manual integration sanity check. Returns 0 on
    a captured rising edge, 1 on timeout, 2 if the trigger is disabled.
    Unlike :func:`run_self_test`, this *does* talk to the live serial
    bus, so it is opt-in via ``--self-test --wait-for-trigger N``.
    """
    import time

    from inspection_system.app.camera_interface import load_config
    from inspection_system.app.io.input_trigger import build_input_trigger

    config = load_config()
    trigger = build_input_trigger(config)
    enabled = bool(getattr(trigger, "enabled", False))
    print(
        f"[wait-for-trigger] trigger={type(trigger).__name__} enabled={enabled}"
    )
    if not enabled:
        last_err = getattr(trigger, "_last_error", None)
        if last_err:
            print(f"  trigger disabled: {last_err}")
        else:
            print("  trigger is not configured (io.trigger.enabled=false).")
        try:
            trigger.cleanup()
        except Exception:
            pass
        return 2

    print(f"[wait-for-trigger] press your switch within {seconds:.1f}s ...", flush=True)
    period = 0.02  # 50 Hz, matches production_screen
    deadline = time.monotonic() + seconds
    edges = 0
    polls = 0
    try:
        while time.monotonic() < deadline:
            try:
                edges = trigger.poll()
            except Exception as exc:
                print(f"  poll error: {exc}")
                edges = 0
            polls += 1
            if edges:
                elapsed = seconds - (deadline - time.monotonic())
                print(f"[wait-for-trigger] EDGE caught after {elapsed:.2f}s, {polls} polls.")
                return 0
            time.sleep(period)
    finally:
        try:
            trigger.cleanup()
        except Exception:
            pass
    print(f"[wait-for-trigger] TIMEOUT after {seconds:.1f}s ({polls} polls, no edge).")
    return 1


def run_self_test() -> int:
    print("inspection_system self-test")
    print("=" * 40)
    checks = [
        ("config-load", _check_config_loads),
        ("indicator-bus", _check_indicator_bus),
        ("camera-backend (mock)", _check_mock_camera_backend),
        ("pilot-readiness (informational)", _check_pilot_readiness),
    ]
    failed = 0
    for label, fn in checks:
        _label, ok, detail = _check(label, fn)
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] {label}: {detail}")
        if not ok:
            failed += 1
    print("=" * 40)
    if failed:
        print(f"self-test FAILED ({failed} check(s) failed)")
        return 1
    print("self-test PASSED")
    return 0


def run_explain(key: str) -> int:
    """Print the help entry for ``key`` (or all entries when key == 'all').

    Returns 0 on success and 1 when the key is unknown so the caller can
    distinguish a typo from a clean lookup in scripts.
    """
    from inspection_system.app.config_help import (
        all_entries,
        format_all,
        format_entry,
        get_entry,
    )

    requested = (key or "").strip()
    if requested.lower() == "all":
        print(format_all())
        return 0
    entry = get_entry(requested)
    if entry is None:
        print(f"Unknown config key: {requested!r}")
        print("Known keys:")
        for e in all_entries():
            print(f"  - {e.key}")
        return 1
    print(format_entry(entry))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m inspection_system")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the offline smoke test (config, indicator bus, camera, pilot status).",
    )
    parser.add_argument(
        "--wait-for-trigger",
        type=float,
        metavar="SECONDS",
        default=None,
        help=(
            "Used with --self-test: after the offline smoke test passes, "
            "watch the configured input trigger for SECONDS and exit 0 "
            "on the first rising edge, 1 on timeout, 2 if disabled. "
            "Talks to live hardware."
        ),
    )
    parser.add_argument(
        "--explain",
        metavar="KEY",
        help=(
            "Print operator/engineer guidance for a config key (e.g. "
            "'inspection.threshold_value') or 'all' to print every entry."
        ),
    )
    args = parser.parse_args(argv)
    if args.self_test:
        rc = run_self_test()
        if rc != 0:
            return rc
        if args.wait_for_trigger is not None:
            return _wait_for_trigger(float(args.wait_for_trigger))
        return 0
    if args.explain is not None:
        return run_explain(args.explain)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    sys.exit(main())

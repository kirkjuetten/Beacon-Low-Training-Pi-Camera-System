"""Camera backend abstraction.

Background
----------
The inspection system historically called :func:`capture_to_temp` directly
from eight call sites, each unpacking a ``(return_code, image_path, stderr)``
tuple and reimplementing minor amounts of error handling. The capture itself
shells out to ``rpicam-still`` on the Pi, with a polled file lock to avoid
concurrent invocations clobbering the same temp file.

That arrangement had three concrete failure modes:

1. If a previous invocation crashed mid-capture, the ``.lock`` file could be
   left on disk; subsequent captures would block for ``max_wait`` seconds and
   then fail.
2. Latency was invisible: there was no log of how long ``rpicam-still`` took.
3. Swapping the implementation (mock for tests, ``picamera2`` later) required
   touching all eight call sites.

This module fixes all three with a small abstraction:

* :class:`CaptureResult` is a dataclass replacing the bare 3-tuple. It also
  carries ``latency_ms`` and an ``error_kind`` for structured logging.
* :class:`CameraBackend` is a :class:`typing.Protocol` covering the surface
  every backend must implement.
* :class:`RpicamStillBackend` reproduces the original ``rpicam-still`` flow
  with hardened lock handling: locks record ``pid`` and a UTC timestamp, and
  any lock older than ``2 * max_wait_s`` is reaped automatically. The active
  lock is registered with :mod:`atexit` so a clean process shutdown leaves no
  residue. The legacy free functions in
  :mod:`inspection_system.app.frame_acquisition` now delegate to a shared
  ``RpicamStillBackend`` instance, so existing callers keep their current
  signatures while gaining the resilience improvements.
* :class:`MockCameraBackend` makes host-side unit tests and the upcoming
  Phase 3 self-test deterministic.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


# --- public data types ------------------------------------------------------


@dataclass
class CaptureResult:
    """Outcome of a single capture call.

    Attributes
    ----------
    return_code:
        ``0`` on success; non-zero on any failure. Mirrors the historical
        :func:`subprocess.run` exit code so legacy callers can keep their
        ``if result_code != 0`` checks.
    image_path:
        Where the captured frame was written. Always populated, even on
        failure, because some callers use it for cleanup.
    stderr:
        Captured stderr text (trimmed). Empty string when not applicable.
    latency_ms:
        Wall-clock milliseconds spent inside the backend's ``capture`` call,
        including lock acquisition and subprocess execution.
    error_kind:
        Coarse category for structured logging:
        ``"ok"``, ``"locked"``, ``"timeout"``, ``"subprocess_error"``,
        ``"missing_output"``, ``"unknown"``.
    """

    return_code: int
    image_path: Path
    stderr: str = ""
    latency_ms: float = 0.0
    error_kind: str = "ok"

    @property
    def ok(self) -> bool:
        return self.return_code == 0


@runtime_checkable
class CameraBackend(Protocol):
    """Minimum surface every camera backend must support."""

    name: str

    def capture(self, config: dict) -> CaptureResult: ...
    def cleanup(self) -> None: ...


# --- shared lock helpers ----------------------------------------------------


def _write_lock_file(lock_path: Path) -> None:
    """Write a lock file describing the current process and timestamp."""
    payload = {"pid": os.getpid(), "ts": time.time()}
    lock_path.write_text(json.dumps(payload), encoding="utf-8")


def _read_lock_file(lock_path: Path) -> Optional[dict]:
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lock_is_stale(lock_path: Path, max_age_s: float) -> bool:
    """Return True if the lock looks abandoned and should be reaped.

    A lock is considered stale when:

    * it is missing required metadata (older format / corrupted), or
    * its timestamp is more than ``max_age_s`` seconds in the past.

    Dead-PID detection deliberately is *not* used: it varies by platform and
    is unreliable across process reuse. Age-based reaping is sufficient
    because the legitimate lock holder is bounded by the capture timeout.
    """
    payload = _read_lock_file(lock_path)
    if payload is None:
        return True
    ts = payload.get("ts")
    if not isinstance(ts, (int, float)):
        return True
    return (time.time() - float(ts)) > max_age_s


def _reap_if_stale(lock_path: Path, max_age_s: float) -> bool:
    """Remove a stale lock file if present. Returns True if it was reaped."""
    if not lock_path.exists():
        return False
    if not _lock_is_stale(lock_path, max_age_s):
        return False
    try:
        lock_path.unlink()
    except OSError:
        return False
    return True


# --- backends ---------------------------------------------------------------


@dataclass
class RpicamStillBackend:
    """Production backend: shells out to ``rpicam-still``.

    Replicates the previous behavior in
    :mod:`inspection_system.app.frame_acquisition` so callers can keep their
    existing flow, plus:

    * Lock files now contain JSON with ``pid`` and ``ts`` (Unix time).
    * Stale locks (older than ``2 * max_wait_s``) are reaped automatically
      both when ``capture`` is entered and at process exit via :mod:`atexit`.
    * Wall-clock latency is measured and surfaced via :class:`CaptureResult`.
    """

    temp_image: Path
    timeout_s: float = 30.0
    max_wait_s: float = 10.0
    poll_interval_s: float = 0.1
    name: str = "rpicam-still"

    _atexit_registered: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Reap any stale lock left behind by a prior crashed run.
        _reap_if_stale(self._lock_path(), max_age_s=2.0 * self.max_wait_s)
        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

    def capture(self, config: dict) -> CaptureResult:
        from inspection_system.app.camera_interface import build_capture_command

        started = time.perf_counter()
        self._cleanup_temp_image()

        lock_path = self._lock_path()
        # Reap a stale lock up front so a crashed prior run can't block us.
        _reap_if_stale(lock_path, max_age_s=2.0 * self.max_wait_s)

        wait_started = time.time()
        while lock_path.exists() and (time.time() - wait_started) < self.max_wait_s:
            if _reap_if_stale(lock_path, max_age_s=2.0 * self.max_wait_s):
                break
            time.sleep(self.poll_interval_s)

        if lock_path.exists():
            return CaptureResult(
                return_code=1,
                image_path=self.temp_image,
                stderr="Temp file locked by another process",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error_kind="locked",
            )

        try:
            _write_lock_file(lock_path)
            cmd = build_capture_command(config, self.temp_image)
            print("Capturing image...")
            print("Command:", " ".join(cmd))
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                print(f"Camera capture timed out after {self.timeout_s} seconds")
                return CaptureResult(
                    return_code=1,
                    image_path=self.temp_image,
                    stderr="Camera capture timeout",
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    error_kind="timeout",
                )
            stderr_text = (result.stderr or "").strip()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            kind = "ok" if result.returncode == 0 else "subprocess_error"
            return CaptureResult(
                return_code=result.returncode,
                image_path=self.temp_image,
                stderr=stderr_text,
                latency_ms=elapsed_ms,
                error_kind=kind,
            )
        finally:
            if lock_path.exists():
                try:
                    lock_path.unlink()
                except OSError:
                    pass

    def cleanup(self) -> None:
        self._cleanup_temp_image()
        self._atexit_cleanup()

    # --- internals ----------------------------------------------------------

    def _lock_path(self) -> Path:
        return self.temp_image.with_suffix(".lock")

    def _cleanup_temp_image(self) -> None:
        if self.temp_image.exists():
            try:
                self.temp_image.unlink()
            except OSError:
                pass

    def _atexit_cleanup(self) -> None:
        # Only remove a lock that this process owns; otherwise rely on the
        # age-based stale check the next caller will perform.
        lock_path = self._lock_path()
        payload = _read_lock_file(lock_path)
        if payload is None:
            return
        if payload.get("pid") == os.getpid():
            try:
                lock_path.unlink()
            except OSError:
                pass


@dataclass
class MockCameraBackend:
    """Deterministic backend used by tests and the host self-test path.

    Each call copies ``source_image`` to ``temp_image``. If ``source_image``
    is ``None``, an empty file is created (useful for exercising failure
    handling). Set ``return_code`` and ``stderr`` to simulate failures.
    """

    temp_image: Path
    source_image: Optional[Path] = None
    return_code: int = 0
    stderr: str = ""
    latency_ms: float = 1.0
    name: str = "mock"

    def capture(self, config: dict) -> CaptureResult:
        if self.return_code == 0:
            if self.source_image is not None and self.source_image.exists():
                shutil.copyfile(self.source_image, self.temp_image)
            else:
                self.temp_image.write_bytes(b"")
            return CaptureResult(
                return_code=0,
                image_path=self.temp_image,
                stderr="",
                latency_ms=self.latency_ms,
                error_kind="ok",
            )
        return CaptureResult(
            return_code=self.return_code,
            image_path=self.temp_image,
            stderr=self.stderr,
            latency_ms=self.latency_ms,
            error_kind="subprocess_error",
        )

    def cleanup(self) -> None:
        if self.temp_image.exists():
            try:
                self.temp_image.unlink()
            except OSError:
                pass

"""Tests for inspection_system.app.camera.backend."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from unittest import mock

import pytest

from inspection_system.app.camera.backend import (
    CameraBackend,
    CaptureResult,
    MockCameraBackend,
    RpicamStillBackend,
    _lock_is_stale,
    _reap_if_stale,
)


# --- CaptureResult ----------------------------------------------------------


def test_capture_result_ok_property():
    assert CaptureResult(return_code=0, image_path=Path("x")).ok is True
    assert CaptureResult(return_code=1, image_path=Path("x")).ok is False


# --- MockCameraBackend ------------------------------------------------------


def test_mock_backend_writes_temp_image_and_satisfies_protocol(tmp_path):
    src = tmp_path / "src.jpg"
    src.write_bytes(b"hello-jpeg")
    dest = tmp_path / "temp.jpg"

    backend = MockCameraBackend(temp_image=dest, source_image=src)
    assert isinstance(backend, CameraBackend)
    result = backend.capture({})

    assert result.ok is True
    assert result.error_kind == "ok"
    assert dest.read_bytes() == b"hello-jpeg"
    assert result.latency_ms > 0


def test_mock_backend_simulates_failure(tmp_path):
    backend = MockCameraBackend(
        temp_image=tmp_path / "temp.jpg",
        return_code=2,
        stderr="boom",
    )
    result = backend.capture({})
    assert result.ok is False
    assert result.return_code == 2
    assert result.stderr == "boom"
    assert result.error_kind == "subprocess_error"


# --- RpicamStillBackend lock semantics --------------------------------------


def _make_backend(tmp_path: Path, **overrides) -> RpicamStillBackend:
    return RpicamStillBackend(
        temp_image=tmp_path / "temp.jpg",
        timeout_s=overrides.get("timeout_s", 5.0),
        max_wait_s=overrides.get("max_wait_s", 0.5),
        poll_interval_s=overrides.get("poll_interval_s", 0.01),
    )


def test_lock_is_stale_for_old_or_corrupt_file(tmp_path):
    lock_path = tmp_path / "temp.lock"

    # Missing -> not stale (caller checks existence first); we focus on
    # corrupt and old payloads which the function does evaluate.
    lock_path.write_text("not-json")
    assert _lock_is_stale(lock_path, max_age_s=10.0) is True

    lock_path.write_text(json.dumps({"pid": os.getpid(), "ts": time.time() - 999.0}))
    assert _lock_is_stale(lock_path, max_age_s=10.0) is True

    lock_path.write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}))
    assert _lock_is_stale(lock_path, max_age_s=10.0) is False


def test_reap_if_stale_removes_old_lock(tmp_path):
    lock_path = tmp_path / "temp.lock"
    lock_path.write_text(json.dumps({"pid": 1, "ts": time.time() - 999.0}))
    assert _reap_if_stale(lock_path, max_age_s=10.0) is True
    assert not lock_path.exists()


def test_reap_if_stale_leaves_fresh_lock(tmp_path):
    lock_path = tmp_path / "temp.lock"
    payload = json.dumps({"pid": os.getpid(), "ts": time.time()})
    lock_path.write_text(payload)
    assert _reap_if_stale(lock_path, max_age_s=10.0) is False
    assert lock_path.exists()


def test_capture_returns_locked_when_active_lock_blocks_full_wait(tmp_path):
    backend = _make_backend(tmp_path, max_wait_s=0.1, poll_interval_s=0.01)
    # Pre-create a fresh lock that the stale-reaper will not touch.
    lock_path = backend._lock_path()
    lock_path.write_text(json.dumps({"pid": os.getpid(), "ts": time.time()}))

    result = backend.capture({})
    assert result.return_code == 1
    assert result.error_kind == "locked"
    assert "locked" in result.stderr.lower()
    # The lock must remain because we did not own this attempt's slot.
    assert lock_path.exists()
    lock_path.unlink()


def test_capture_reaps_stale_lock_and_proceeds(tmp_path):
    backend = _make_backend(tmp_path, max_wait_s=0.5)
    lock_path = backend._lock_path()
    # Stale lock from a previous crashed run.
    lock_path.write_text(json.dumps({"pid": 999_999, "ts": time.time() - 999.0}))

    fake_completed = mock.Mock()
    fake_completed.returncode = 0
    fake_completed.stderr = ""
    with mock.patch(
        "inspection_system.app.camera.backend.subprocess.run",
        return_value=fake_completed,
    ):
        result = backend.capture({})

    assert result.return_code == 0
    assert result.error_kind == "ok"
    assert result.latency_ms >= 0
    assert not lock_path.exists()


def test_capture_records_latency_on_timeout(tmp_path):
    backend = _make_backend(tmp_path, timeout_s=0.05)

    def _raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="rpicam-still", timeout=0.05)

    with mock.patch(
        "inspection_system.app.camera.backend.subprocess.run",
        side_effect=_raise_timeout,
    ):
        result = backend.capture({})

    assert result.return_code == 1
    assert result.error_kind == "timeout"
    assert "timeout" in result.stderr.lower()
    assert result.latency_ms >= 0
    assert not backend._lock_path().exists()


def test_capture_releases_lock_on_subprocess_error(tmp_path):
    backend = _make_backend(tmp_path)
    fake_completed = mock.Mock()
    fake_completed.returncode = 5
    fake_completed.stderr = "device busy"
    with mock.patch(
        "inspection_system.app.camera.backend.subprocess.run",
        return_value=fake_completed,
    ):
        result = backend.capture({})

    assert result.return_code == 5
    assert result.error_kind == "subprocess_error"
    assert "device busy" in result.stderr
    assert not backend._lock_path().exists()


# --- frame_acquisition shim back-compat ------------------------------------


def _patched_backend(tmp_path: Path, monkeypatch) -> RpicamStillBackend:
    """Swap the frame_acquisition module-level backend for one in tmp_path.

    The default backend points at ``inspection_system/temp_capture.jpg`` whose
    parent may not exist on CI runners; tests must not depend on that path.
    """
    from inspection_system.app import frame_acquisition

    backend = RpicamStillBackend(temp_image=tmp_path / "temp.jpg")
    monkeypatch.setattr(frame_acquisition, "_default_backend", backend)
    return backend


def test_frame_acquisition_capture_to_temp_returns_legacy_tuple(tmp_path, monkeypatch):
    """The legacy 3-tuple signature must still hold for existing callers."""
    from inspection_system.app import frame_acquisition

    _patched_backend(tmp_path, monkeypatch)
    fake_completed = mock.Mock()
    fake_completed.returncode = 0
    fake_completed.stderr = ""
    monkeypatch.setattr(
        "inspection_system.app.camera.backend.subprocess.run",
        mock.Mock(return_value=fake_completed),
    )

    return_code, image_path, stderr = frame_acquisition.capture_to_temp({})
    assert return_code == 0
    assert isinstance(image_path, Path)
    assert isinstance(stderr, str)


def test_frame_acquisition_capture_frame_returns_capture_result(tmp_path, monkeypatch):
    from inspection_system.app import frame_acquisition

    _patched_backend(tmp_path, monkeypatch)
    fake_completed = mock.Mock()
    fake_completed.returncode = 0
    fake_completed.stderr = ""
    monkeypatch.setattr(
        "inspection_system.app.camera.backend.subprocess.run",
        mock.Mock(return_value=fake_completed),
    )

    result = frame_acquisition.capture_frame({})
    assert isinstance(result, CaptureResult)
    assert result.ok is True
    assert result.latency_ms >= 0

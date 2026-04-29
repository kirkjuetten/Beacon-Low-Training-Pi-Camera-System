"""Tests for the python -m inspection_system self-test entrypoint."""
from __future__ import annotations

from unittest import mock

import pytest

from inspection_system import __main__ as self_test_main


def test_self_test_returns_zero_when_all_checks_pass(capsys, monkeypatch):
    # All four checks succeed against the real (host-side) implementations.
    rc = self_test_main.main(["--self-test"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "self-test PASSED" in out
    for label in (
        "config-load",
        "indicator-bus",
        "camera-backend",
        "pilot-readiness",
    ):
        assert label in out


def test_self_test_returns_one_when_a_check_fails(capsys, monkeypatch):
    # Force the camera-backend check to raise so we hit the FAIL path.
    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(self_test_main, "_check_mock_camera_backend", _boom)
    rc = self_test_main.main(["--self-test"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "self-test FAILED" in out
    assert "[FAIL] camera-backend" in out
    assert "simulated failure" in out


def test_main_without_args_prints_help(capsys):
    rc = self_test_main.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--self-test" in out

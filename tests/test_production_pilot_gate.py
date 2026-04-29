"""Tests for the supervised pilot hard gate in run_production_mode."""
from __future__ import annotations

from unittest import mock

import pytest

from inspection_system.app import production_screen


def test_pilot_gate_enforced_default_is_true():
    assert production_screen._pilot_gate_enforced({}) is True
    assert production_screen._pilot_gate_enforced({"pilot_readiness": {}}) is True


def test_pilot_gate_enforced_false_when_explicitly_disabled():
    config = {"pilot_readiness": {"enforce": False}}
    assert production_screen._pilot_gate_enforced(config) is False


def test_run_production_mode_blocks_when_not_pilot_ready(capsys, monkeypatch):
    # Pretend pygame is available so the only blocker is the pilot gate.
    monkeypatch.setattr(production_screen, "PYGAME_AVAILABLE", True)
    fake_status = {
        "ready": False,
        "current_project": "demo",
        "reference_candidate_count": 0,
        "commissioning": {"summary_line": "needs work"},
        "dataset": {"root": "/tmp", "session_count": 0, "record_count": 0, "split_counts": {}},
        "policy": {"targets": {}},
        "issues": ["No reference"],
        "actions": ["Capture reference"],
        "warnings": [],
        "phases": [],
        "manual_floor_gates": [],
    }
    monkeypatch.setattr(
        "inspection_system.app.pilot_readiness.build_supervised_pilot_status",
        lambda *_a, **_k: fake_status,
    )
    # Sentinel: if the gate fails open, this would be invoked and we want to
    # see the test fail loudly rather than silently launching pygame.
    monkeypatch.setattr(
        "inspection_system.app.inspection_runtime_context.build_inspection_runtime_context",
        mock.Mock(side_effect=AssertionError("gate let production launch through")),
    )

    rc = production_screen.run_production_mode({}, indicator=mock.Mock())
    out = capsys.readouterr().out
    assert rc == 1
    assert "Production launch blocked" in out
    assert "No reference" in out
    assert "pilot_readiness.enforce" in out


def test_run_production_mode_warns_when_gate_overridden(capsys, monkeypatch):
    monkeypatch.setattr(production_screen, "PYGAME_AVAILABLE", True)

    # Stop right after the gate by short-circuiting the runtime context build.
    monkeypatch.setattr(
        production_screen,
        "build_inspection_runtime_context",
        mock.Mock(side_effect=RuntimeError("stop here")),
    )
    pilot_call = mock.Mock()
    monkeypatch.setattr(
        "inspection_system.app.pilot_readiness.build_supervised_pilot_status",
        pilot_call,
    )

    config = {"pilot_readiness": {"enforce": False}}
    with pytest.raises(RuntimeError, match="stop here"):
        production_screen.run_production_mode(config, indicator=mock.Mock())
    out = capsys.readouterr().out
    assert "pilot_readiness.enforce=false" in out
    # When the gate is bypassed we must NOT call build_supervised_pilot_status.
    pilot_call.assert_not_called()

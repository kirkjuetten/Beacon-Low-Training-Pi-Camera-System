"""Tests for inspection_system.app.io.input_trigger."""
from __future__ import annotations

import sys
import time
import types
from typing import Sequence
from unittest import mock

import pytest

from inspection_system.app.io.input_trigger import (
    InputTrigger,
    ModbusInputTrigger,
    NullInputTrigger,
    build_input_trigger,
)


# --- helpers -----------------------------------------------------------------


class _FakeResponse:
    def __init__(self, bits: Sequence[bool], error: bool = False):
        self.bits = list(bits)
        self._err = error

    def isError(self) -> bool:
        return self._err


class _ScriptedClient:
    """Pymodbus-shaped fake whose ``read_discrete_inputs`` walks a script."""

    def __init__(self, bit_script: list[Sequence[bool]]):
        self.script = list(bit_script)
        self.calls = 0
        self.closed = False

    def read_discrete_inputs(self, address, count=1, slave=1):
        if not self.script:
            return _FakeResponse([False] * count)
        bits = self.script.pop(0)
        self.calls += 1
        return _FakeResponse(bits)

    def close(self):
        self.closed = True


def _trigger(client: _ScriptedClient, **overrides) -> ModbusInputTrigger:
    kwargs = dict(
        port="/dev/null",
        slave_id=1,
        register=0x0000,
        channel=0,
        count=8,
        debounce_ms=0,  # default off for unit tests; debounce is exercised separately
        client_factory=lambda: client,
    )
    kwargs.update(overrides)
    return ModbusInputTrigger(**kwargs)


# --- protocol + null implementation ------------------------------------------


def test_null_trigger_satisfies_protocol_and_returns_zero():
    t = NullInputTrigger()
    assert isinstance(t, InputTrigger)
    assert t.enabled is False
    assert t.poll() == 0
    assert t.poll() == 0
    t.cleanup()
    assert t.history == ["poll", "poll", "cleanup"]


def test_build_input_trigger_returns_null_when_disabled():
    config = {"io": {"trigger": {"enabled": False}}}
    t = build_input_trigger(config)
    assert isinstance(t, NullInputTrigger)


def test_build_input_trigger_returns_null_when_block_missing():
    t = build_input_trigger({"io": {}})
    assert isinstance(t, NullInputTrigger)


# --- happy-path edge detection -----------------------------------------------


def test_modbus_trigger_emits_one_edge_on_rising_transition():
    # low -> high -> low -> high : two distinct rising edges.
    script = [
        [False] * 8,
        [True] + [False] * 7,
        [False] * 8,
        [True] + [False] * 7,
    ]
    client = _ScriptedClient(script)
    trig = _trigger(client)

    assert trig.enabled is True
    edges = [trig.poll() for _ in range(4)]
    assert edges == [0, 1, 0, 1]


def test_modbus_trigger_does_not_emit_while_held_high():
    # A long press: one rising edge, then nothing while held.
    script = [[False] * 8] + [[True] + [False] * 7 for _ in range(5)]
    client = _ScriptedClient(script)
    trig = _trigger(client)

    edges = [trig.poll() for _ in range(6)]
    assert edges == [0, 1, 0, 0, 0, 0]


def test_modbus_trigger_uses_correct_channel_offset():
    # Press is on channel 3, not 0.
    script = [
        [False] * 8,
        [False, False, False, True, False, False, False, False],
        [False] * 8,
    ]
    client = _ScriptedClient(script)
    trig = _trigger(client, channel=3)
    assert [trig.poll() for _ in range(3)] == [0, 1, 0]


# --- debounce ----------------------------------------------------------------


def test_modbus_trigger_debounces_a_glitch():
    # A 1-poll spike that doesn't last >= debounce window must NOT fire.
    script = [
        [False] * 8,
        [True] + [False] * 7,    # spike
        [False] * 8,             # fell back before debounce window cleared
        [False] * 8,
    ]
    client = _ScriptedClient(script)
    # 50 ms debounce; we'll poll faster than that with frozen time.
    trig = _trigger(client, debounce_ms=50)

    fake_now = [1000.0]
    with mock.patch(
        "inspection_system.app.io.input_trigger.time.monotonic",
        side_effect=lambda: fake_now[0],
    ):
        # poll #1 at t=0: low (stable)
        assert trig.poll() == 0
        fake_now[0] += 0.01
        # poll #2 at t=10ms: high (candidate, not stable yet)
        assert trig.poll() == 0
        fake_now[0] += 0.01
        # poll #3 at t=20ms: dropped back low before debounce window elapsed
        assert trig.poll() == 0
        fake_now[0] += 1.0
        # poll #4 at t=1.02s: still low (final stable state, no edge fired)
        assert trig.poll() == 0


def test_modbus_trigger_fires_when_debounce_window_satisfied():
    script = [
        [False] * 8,
        [True] + [False] * 7,
        [True] + [False] * 7,
        [True] + [False] * 7,
    ]
    client = _ScriptedClient(script)
    trig = _trigger(client, debounce_ms=50)

    fake_now = [1000.0]
    with mock.patch(
        "inspection_system.app.io.input_trigger.time.monotonic",
        side_effect=lambda: fake_now[0],
    ):
        # baseline low
        assert trig.poll() == 0
        fake_now[0] += 0.01
        # candidate high established at +10ms
        assert trig.poll() == 0
        fake_now[0] += 0.06
        # +70ms total: still high, debounce window elapsed -> edge fires
        assert trig.poll() == 1
        fake_now[0] += 0.05
        # held high: no further edges
        assert trig.poll() == 0


# --- error paths -------------------------------------------------------------


def test_modbus_trigger_disabled_when_channel_out_of_range():
    trig = ModbusInputTrigger(
        port="/dev/null",
        channel=8,
        count=8,
        client_factory=lambda: _ScriptedClient([]),
    )
    assert trig.enabled is False
    assert trig._last_error is not None and "channel" in trig._last_error
    assert trig.poll() == 0


def test_modbus_trigger_handles_modbus_error_response():
    class _ErrClient:
        def read_discrete_inputs(self, *a, **kw):
            return _FakeResponse([False] * 8, error=True)

        def close(self):
            pass

    trig = _trigger(_ErrClient())
    assert trig.poll() == 0
    assert "modbus error" in (trig._last_error or "")


def test_modbus_trigger_falls_back_to_unit_kwarg_for_pymodbus_2x():
    """pymodbus 2.x uses ``unit=`` instead of ``slave=``; we must not break."""
    last_kwargs: dict[str, object] = {}

    class _PymodbusV2Client:
        def read_discrete_inputs(self, address, count=1, **kwargs):
            last_kwargs.clear()
            last_kwargs.update(kwargs)
            if "slave" in kwargs:
                raise TypeError("got unexpected keyword 'slave'")
            return _FakeResponse([True] + [False] * 7)

        def close(self):
            pass

    trig = _trigger(_PymodbusV2Client())
    edges = trig.poll()
    assert edges == 1
    # The successful call used `unit=` after the `slave=` TypeError fallback.
    assert last_kwargs == {"unit": 1}


def test_modbus_trigger_disabled_when_pymodbus_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymodbus", None)
    trig = ModbusInputTrigger(port="/dev/null", channel=0)
    assert trig.enabled is False
    assert trig._last_error is not None and "pymodbus" in trig._last_error
    # Calls become no-ops; must not raise.
    assert trig.poll() == 0
    trig.cleanup()


# --- factory + cleanup -------------------------------------------------------


def test_build_input_trigger_reads_config(monkeypatch):
    fake_module = types.ModuleType("pymodbus")
    fake_client_module = types.ModuleType("pymodbus.client")

    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def connect(self):
            return True

        def read_discrete_inputs(self, *a, **kw):
            return _FakeResponse([False] * 8)

        def close(self):
            pass

    fake_client_module.ModbusSerialClient = _FakeClient
    fake_module.client = fake_client_module
    monkeypatch.setitem(sys.modules, "pymodbus", fake_module)
    monkeypatch.setitem(sys.modules, "pymodbus.client", fake_client_module)

    config = {
        "io": {
            "modbus": {"port": "/dev/ttyFOO"},
            "trigger": {
                "enabled": True,
                "slave_id": 5,
                "channel": 2,
                "register": 0x0010,
                "count": 4,
                "debounce_ms": 75,
                "timeout_s": 0.5,
            },
        }
    }
    trig = build_input_trigger(config)
    assert isinstance(trig, ModbusInputTrigger)
    assert trig.enabled is True
    assert trig.slave_id == 5
    assert trig.channel == 2
    assert trig.register == 0x0010
    assert trig.count == 4
    assert trig.debounce_ms == 75
    # Inherits port from io.modbus.port when trigger.port is unset.
    assert captured.get("port") == "/dev/ttyFOO"


def test_modbus_trigger_cleanup_closes_client():
    client = _ScriptedClient([])
    trig = _trigger(client)
    trig.cleanup()
    assert client.closed is True
    # Second cleanup is a no-op.
    trig.cleanup()


def test_modbus_trigger_health_check_reports_state():
    client = _ScriptedClient([])
    trig = _trigger(client)
    health = trig.health_check()
    assert health["enabled"] is True
    assert health["channel"] == 0
    assert health["slave_id"] == 1

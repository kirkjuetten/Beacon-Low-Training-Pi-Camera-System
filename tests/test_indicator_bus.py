"""Tests for inspection_system.app.io.indicator_bus."""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest import mock

import pytest

from inspection_system.app.io.indicator_bus import (
    IndicatorBus,
    ModbusIndicatorBus,
    NullIndicatorBus,
    build_indicator_bus,
)


def test_null_bus_records_calls_and_satisfies_protocol():
    bus = NullIndicatorBus()
    assert isinstance(bus, IndicatorBus)
    assert bus.enabled is False

    bus.pulse_pass()
    bus.pulse_fail()
    bus.cleanup()
    assert bus.history == ["pass", "fail", "cleanup"]


def test_build_indicator_bus_dispatches_none_mode():
    bus = build_indicator_bus({"io": {"mode": "none"}})
    assert isinstance(bus, NullIndicatorBus)


def test_build_indicator_bus_default_mode_uses_legacy_indicator_led():
    # No "io" block at all -> legacy IndicatorLED via indicator_context.
    config = {"indicator_led": {"enabled": False, "pass_gpio": 23, "fail_gpio": 24, "pulse_ms": 750}}
    bus = build_indicator_bus(config)
    # Legacy IndicatorLED satisfies the Protocol structurally.
    assert hasattr(bus, "pulse_pass") and hasattr(bus, "pulse_fail") and hasattr(bus, "cleanup")
    assert bus.enabled is False  # GPIO disabled in this test config


def _install_fake_pymodbus(monkeypatch, write_coil_impl):
    """Install a fake `pymodbus.client.ModbusSerialClient` and return the mock."""
    fake_module = types.ModuleType("pymodbus")
    fake_client_module = types.ModuleType("pymodbus.client")

    instances: list[Any] = []

    class _FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.connected = False
            self.write_coil = mock.Mock(side_effect=write_coil_impl)
            self.close = mock.Mock()
            instances.append(self)

        def connect(self):
            self.connected = True
            return True

    fake_client_module.ModbusSerialClient = _FakeClient
    fake_module.client = fake_client_module
    monkeypatch.setitem(sys.modules, "pymodbus", fake_module)
    monkeypatch.setitem(sys.modules, "pymodbus.client", fake_client_module)
    return instances


def _ok_response():
    response = mock.Mock()
    response.isError.return_value = False
    return response


def test_modbus_bus_pulse_pass_writes_expected_coils(monkeypatch):
    instances = _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())

    bus = ModbusIndicatorBus(port="/dev/ttyUSB0", pass_channel=0, fail_channel=1, pulse_ms=1)
    assert bus.enabled is True

    with mock.patch("inspection_system.app.io.indicator_bus.time.sleep") as sleep_mock:
        bus.pulse_pass()
        bus.cleanup()

    client = instances[0]
    # Expect: clear pass, clear fail, set pass HIGH, sleep, clear pass; then cleanup clears both.
    coil_calls = [c.args[:2] for c in client.write_coil.call_args_list]
    assert coil_calls[:5] == [(0, False), (1, False), (0, True), (0, False), (0, False)]
    sleep_mock.assert_called()  # we slept during the pulse


def test_modbus_bus_disables_when_pymodbus_missing(monkeypatch):
    # Ensure import fails by removing pymodbus from sys.modules and blocking import.
    monkeypatch.setitem(sys.modules, "pymodbus", None)
    bus = ModbusIndicatorBus(port="/dev/ttyUSB0")
    assert bus.enabled is False
    assert bus._last_error is not None and "pymodbus" in bus._last_error
    # Calls become no-ops; must not raise.
    bus.pulse_pass()
    bus.pulse_fail()
    bus.cleanup()


def test_modbus_bus_health_check_reports_state(monkeypatch):
    _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())
    bus = ModbusIndicatorBus(port="/dev/ttyUSB0")
    health = bus.health_check()
    assert health["enabled"] is True
    assert health["port"] == "/dev/ttyUSB0"
    assert health["last_error"] is None


def test_build_indicator_bus_modbus_mode_uses_modbus_bus(monkeypatch):
    _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())
    config = {
        "io": {
            "mode": "modbus",
            "modbus": {"port": "/dev/ttyUSB1", "slave_id": 7, "pass_channel": 2, "fail_channel": 3},
            "relay": {"slave_id": 4, "pass_channel": 0, "fail_channel": 1},
        }
    }
    bus = build_indicator_bus(config)
    assert isinstance(bus, ModbusIndicatorBus)
    assert bus.slave_id == 7
    assert bus.pass_channel == 2
    assert bus.relay is not None and bus.relay.slave_id == 4


def test_modbus_bus_set_outputs_returns_false_when_disabled():
    bus = ModbusIndicatorBus(port="/dev/null", enabled=False)
    assert bus.set_outputs([0, 1, 2], True) is False

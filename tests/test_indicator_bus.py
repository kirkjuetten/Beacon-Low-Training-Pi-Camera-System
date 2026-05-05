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
from inspection_system.app.io.indicator_bus import (
    _build_flash_on_frame,
    _crc16_modbus,
    _ModbusRelayMap,
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

    class _FakeSerialPort:
        def __init__(self):
            self.writes: list[bytes] = []
            self.flushed = 0
            self.input_resets = 0
            self.reads: list[int] = []

        def write(self, data: bytes) -> int:
            self.writes.append(bytes(data))
            return len(data)

        def flush(self) -> None:
            self.flushed += 1

        def reset_input_buffer(self) -> None:
            self.input_resets += 1

        def read(self, size: int) -> bytes:
            # Echo the most recent write back, like a real Modbus 0x05 device.
            self.reads.append(size)
            if self.writes:
                return self.writes[-1][:size]
            return b""

    class _FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.connected = False
            self.write_coil = mock.Mock(side_effect=write_coil_impl)
            self.close = mock.Mock()
            self.socket = _FakeSerialPort()
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


# --- V3 flash-on protocol primitives -----------------------------------------


def test_crc16_matches_known_waveshare_example():
    # From the Waveshare wiki: relay 0 flash-on for 700ms (slave 0x01):
    #   01 05 02 00 00 07 8D B0
    body = bytes.fromhex("01 05 02 00 00 07")
    assert _crc16_modbus(body) == bytes([0x8D, 0xB0])


def test_build_flash_on_frame_matches_known_waveshare_example():
    # slave=1, channel=0, deciseconds=7 (700 ms) -> 01 05 02 00 00 07 8D B0
    frame = _build_flash_on_frame(slave_id=1, channel=0, deciseconds=7)
    assert frame.hex(" ") == "01 05 02 00 00 07 8d b0"


def test_build_flash_on_frame_uses_correct_register_offset():
    # Channel 1 -> register 0x0201.
    frame = _build_flash_on_frame(slave_id=2, channel=1, deciseconds=10)
    assert frame[0] == 0x02
    assert frame[1] == 0x05
    assert frame[2:4] == bytes([0x02, 0x01])
    assert frame[4:6] == bytes([0x00, 0x0A])


@pytest.mark.parametrize("channel", [-1, 8, 99])
def test_build_flash_on_frame_rejects_out_of_range_channel(channel):
    with pytest.raises(ValueError):
        _build_flash_on_frame(slave_id=2, channel=channel, deciseconds=10)


@pytest.mark.parametrize("deciseconds", [0, -1, 0x8000, 0x10000])
def test_build_flash_on_frame_rejects_out_of_range_duration(deciseconds):
    with pytest.raises(ValueError):
        _build_flash_on_frame(slave_id=2, channel=0, deciseconds=deciseconds)


# --- relay-routing pulse behavior --------------------------------------------


def test_modbus_bus_relay_target_writes_flash_on_frame_and_does_not_sleep(monkeypatch):
    instances = _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())
    bus = ModbusIndicatorBus(
        port="/dev/ttyUSB0",
        indicator_target="relay",
        pass_pulse_ms=3000,
        fail_pulse_ms=2500,
        relay=_ModbusRelayMap(slave_id=2, pass_channel=0, fail_channel=1),
    )
    with mock.patch("inspection_system.app.io.indicator_bus.time.sleep") as sleep_mock:
        bus.pulse_pass()
        bus.pulse_fail()

    client = instances[0]
    # Flash-on path must NOT touch write_coil at all (no IO-module coils).
    assert client.write_coil.call_count == 0
    # And it must not block the calling thread.
    sleep_mock.assert_not_called()

    writes = client.socket.writes
    assert len(writes) == 2
    # pass: slave 2, channel 0, 30 deciseconds (3000ms)
    assert writes[0] == _build_flash_on_frame(2, 0, 30)
    # fail: slave 2, channel 1, 25 deciseconds (2500ms)
    assert writes[1] == _build_flash_on_frame(2, 1, 25)


def test_modbus_bus_relay_target_clamps_duration_to_protocol_range(monkeypatch):
    instances = _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())
    bus = ModbusIndicatorBus(
        port="/dev/ttyUSB0",
        indicator_target="relay",
        pass_pulse_ms=10,                  # rounds to 0 -> clamped up to 1 decisecond
        fail_pulse_ms=10_000_000,          # huge -> clamped down to 0x7FFF
        relay=_ModbusRelayMap(slave_id=2, pass_channel=0, fail_channel=1),
    )
    bus.pulse_pass()
    bus.pulse_fail()
    writes = instances[0].socket.writes
    assert writes[0] == _build_flash_on_frame(2, 0, 1)
    assert writes[1] == _build_flash_on_frame(2, 1, 0x7FFF)


def test_modbus_bus_relay_target_records_error_when_relay_not_configured(monkeypatch):
    _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())
    bus = ModbusIndicatorBus(port="/dev/ttyUSB0", indicator_target="relay", relay=None)
    bus.pulse_pass()
    assert bus._last_error is not None
    assert "no relay map" in bus._last_error


def test_modbus_bus_relay_target_records_error_when_channel_unset(monkeypatch):
    _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())
    bus = ModbusIndicatorBus(
        port="/dev/ttyUSB0",
        indicator_target="relay",
        relay=_ModbusRelayMap(slave_id=2, pass_channel=None, fail_channel=1),
    )
    bus.pulse_pass()  # no pass_channel -> error
    assert bus._last_error is not None and "pass_channel" in bus._last_error


def test_modbus_bus_io_module_target_uses_per_pulse_duration_when_set(monkeypatch):
    _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())
    bus = ModbusIndicatorBus(
        port="/dev/ttyUSB0",
        indicator_target="io_module",
        pulse_ms=500,
        pass_pulse_ms=1234,
    )
    with mock.patch("inspection_system.app.io.indicator_bus.time.sleep") as sleep_mock:
        bus.pulse_pass()
    sleep_mock.assert_called_once()
    assert sleep_mock.call_args.args[0] == pytest.approx(1.234)


def test_modbus_bus_cleanup_skips_coil_writes_in_relay_mode(monkeypatch):
    instances = _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())
    bus = ModbusIndicatorBus(
        port="/dev/ttyUSB0",
        indicator_target="relay",
        relay=_ModbusRelayMap(slave_id=2, pass_channel=0, fail_channel=1),
    )
    bus.cleanup()
    # No write_coil during cleanup; relay flash-on auto-clears.
    assert instances[0].write_coil.call_count == 0


def test_modbus_bus_io_module_target_noops_when_pass_channel_unassigned(monkeypatch):
    instances = _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())
    bus = ModbusIndicatorBus(
        port="/dev/ttyUSB0",
        indicator_target="io_module",
        pass_channel=None,
        fail_channel=1,
    )

    with mock.patch("inspection_system.app.io.indicator_bus.time.sleep") as sleep_mock:
        bus.pulse_pass()

    assert instances[0].write_coil.call_count == 0
    sleep_mock.assert_not_called()


def test_build_indicator_bus_reads_relay_routing_and_per_pulse_durations(monkeypatch):
    _install_fake_pymodbus(monkeypatch, write_coil_impl=lambda *a, **kw: _ok_response())
    config = {
        "io": {
            "mode": "modbus",
            "indicator_target": "relay",
            "pulse_ms": 750,
            "pass_pulse_ms": 3000,
            "fail_pulse_ms": 2000,
            "modbus": {"port": "/dev/ttyUSB0"},
            "relay": {"slave_id": 2, "pass_channel": 0, "fail_channel": 1},
        }
    }
    bus = build_indicator_bus(config)
    assert isinstance(bus, ModbusIndicatorBus)
    assert bus.indicator_target == "relay"
    assert bus.pass_pulse_ms == 3000
    assert bus.fail_pulse_ms == 2000
    assert bus.relay is not None and bus.relay.pass_channel == 0

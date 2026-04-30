from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest import mock

from inspection_system.app.io.modbus_session import open_shared_modbus_client


def _install_fake_pymodbus(monkeypatch):
    fake_module = types.ModuleType("pymodbus")
    fake_client_module = types.ModuleType("pymodbus.client")
    instances = []

    class _FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.timeout = kwargs["timeout"]
            self.socket = SimpleNamespace(timeout=kwargs["timeout"], write=lambda _data: None)
            self.close = mock.Mock()
            instances.append(self)

        def connect(self):
            return True

    fake_client_module.ModbusSerialClient = _FakeClient
    fake_module.client = fake_client_module
    monkeypatch.setitem(sys.modules, "pymodbus", fake_module)
    monkeypatch.setitem(sys.modules, "pymodbus.client", fake_client_module)
    return instances


def test_open_shared_modbus_client_reuses_underlying_client(monkeypatch):
    instances = _install_fake_pymodbus(monkeypatch)

    handle_a = open_shared_modbus_client(
        port="/dev/ttySHARED0",
        baudrate=9600,
        parity="N",
        stopbits=1,
        bytesize=8,
        timeout_s=0.3,
    )
    handle_b = open_shared_modbus_client(
        port="/dev/ttySHARED0",
        baudrate=9600,
        parity="N",
        stopbits=1,
        bytesize=8,
        timeout_s=1.0,
    )

    assert len(instances) == 1
    assert handle_a.socket is handle_b.socket
    assert instances[0].timeout == 1.0
    assert instances[0].socket.timeout == 1.0

    handle_a.close()
    assert instances[0].close.call_count == 0
    handle_b.close()
    assert instances[0].close.call_count == 1


def test_open_shared_modbus_client_does_not_share_different_ports(monkeypatch):
    instances = _install_fake_pymodbus(monkeypatch)

    handle_a = open_shared_modbus_client(
        port="/dev/ttySHARED1",
        baudrate=9600,
        parity="N",
        stopbits=1,
        bytesize=8,
        timeout_s=0.3,
    )
    handle_b = open_shared_modbus_client(
        port="/dev/ttySHARED2",
        baudrate=9600,
        parity="N",
        stopbits=1,
        bytesize=8,
        timeout_s=0.3,
    )

    assert len(instances) == 2
    assert handle_a.socket is not handle_b.socket

    handle_a.close()
    handle_b.close()
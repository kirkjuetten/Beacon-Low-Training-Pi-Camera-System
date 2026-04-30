"""Tests for the relay address provisioning helper."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from inspection_system.app.io.provision_relay import (
    _DEVICE_ADDRESS_REGISTER,
    main,
    provision,
)


class _FakeResult:
    def __init__(self, registers=None, error: bool = False):
        self.registers = registers or []
        self._error = error

    def isError(self) -> bool:  # noqa: N802 - mimic pymodbus
        return self._error


@dataclass
class _FakeClient:
    """In-memory stand-in for ModbusSerialClient.

    Tracks a single device whose address starts at ``address`` and exposes
    register 0x4000 holding that same address. Writes to 0x4000 update both.
    """

    address: int
    write_log: list = field(default_factory=list)
    read_log: list = field(default_factory=list)
    write_should_fail: bool = False
    read_should_fail: bool = False

    def connect(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def read_holding_registers(self, address: int, count: int = 1, slave: Optional[int] = None, **_):
        self.read_log.append((address, slave))
        if self.read_should_fail:
            return _FakeResult(error=True)
        if slave != self.address:
            return _FakeResult(error=True)  # device with that address does not exist
        if address == _DEVICE_ADDRESS_REGISTER:
            return _FakeResult(registers=[self.address])
        return _FakeResult(error=True)

    def write_register(self, address: int, value: int, slave: Optional[int] = None, **_):
        self.write_log.append((address, value, slave))
        if self.write_should_fail:
            return _FakeResult(error=True)
        if slave != self.address:
            return _FakeResult(error=True)
        if address == _DEVICE_ADDRESS_REGISTER:
            self.address = value
            return _FakeResult()
        return _FakeResult(error=True)


def test_provision_changes_address_when_device_responds():
    client = _FakeClient(address=1)

    result = provision(client, current_address=1, new_address=2, settle_seconds=0)

    assert result.success
    assert result.before_address == 1
    assert result.after_address == 2
    assert client.address == 2
    assert (_DEVICE_ADDRESS_REGISTER, 2, 1) in client.write_log


def test_provision_fails_cleanly_when_no_device_responds():
    client = _FakeClient(address=99)  # not at 1

    result = provision(client, current_address=1, new_address=2, settle_seconds=0)

    assert not result.success
    assert "no response from slave 1" in result.message
    assert client.address == 99  # untouched


def test_provision_refuses_when_new_equals_current():
    client = _FakeClient(address=1)

    result = provision(client, current_address=1, new_address=1, settle_seconds=0)

    assert not result.success
    assert "nothing to do" in result.message
    assert client.write_log == []


def test_provision_rejects_out_of_range_new_address():
    client = _FakeClient(address=1)

    result = provision(client, current_address=1, new_address=0, settle_seconds=0)

    assert not result.success
    assert "out of range" in result.message
    assert client.write_log == []


def test_provision_reports_write_failure():
    client = _FakeClient(address=1, write_should_fail=True)

    result = provision(client, current_address=1, new_address=2, settle_seconds=0)

    assert not result.success
    assert "write" in result.message.lower()
    assert client.address == 1


def test_main_dry_run_does_not_open_serial(monkeypatch, capsys):
    # If the dry-run path tries to open the serial port, this would explode.
    def boom(*_args, **_kwargs):  # pragma: no cover - fail loudly
        raise AssertionError("serial port must not be opened in dry-run mode")

    monkeypatch.setattr(
        "inspection_system.app.io.provision_relay._open_serial_client", boom
    )

    rc = main(["--port", "/dev/null", "--new-address", "2"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Dry run" in out


def test_main_confirm_path_returns_success(monkeypatch, capsys):
    fake = _FakeClient(address=1)
    monkeypatch.setattr(
        "inspection_system.app.io.provision_relay._open_serial_client",
        lambda *a, **k: fake,
    )

    rc = main(["--port", "/dev/null", "--new-address", "2", "--confirm"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert fake.address == 2


def test_main_confirm_path_reports_no_response(monkeypatch, capsys):
    fake = _FakeClient(address=99)
    monkeypatch.setattr(
        "inspection_system.app.io.provision_relay._open_serial_client",
        lambda *a, **k: fake,
    )

    rc = main(["--port", "/dev/null", "--new-address", "2", "--confirm"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "FAIL" in err

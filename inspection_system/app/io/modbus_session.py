"""Shared Modbus serial-session helpers.

This module lets multiple IO abstractions reuse a single ``pymodbus``
``ModbusSerialClient`` for the same serial port. That prevents the trigger
and relay indicator paths from fighting over ``/dev/ttyUSB0`` while keeping
their public APIs unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class _SessionKey:
    port: str
    baudrate: int
    parity: str
    stopbits: int
    bytesize: int


@dataclass
class _SessionEntry:
    client: object
    refs: int
    timeout_s: float


_SESSION_LOCK = Lock()
_SESSIONS: dict[_SessionKey, _SessionEntry] = {}


def resolve_raw_serial_port(client: object):
    """Return the underlying pyserial port held by a pymodbus client, or None."""
    if client is None:
        return None
    for attr in ("socket", "_socket"):
        port = getattr(client, attr, None)
        if port is not None and hasattr(port, "write"):
            return port
    transport = getattr(client, "transport", None)
    if transport is not None:
        for attr in ("serial", "socket"):
            port = getattr(transport, attr, None)
            if port is not None and hasattr(port, "write"):
                return port
    return None


def _apply_timeout(client: object, timeout_s: float) -> None:
    """Promote the shared client's timeout to at least ``timeout_s``."""
    try:
        current = float(getattr(client, "timeout", timeout_s))
    except Exception:
        current = timeout_s
    promoted = max(current, timeout_s)
    try:
        setattr(client, "timeout", promoted)
    except Exception:
        pass

    port = resolve_raw_serial_port(client)
    if port is not None:
        try:
            port.timeout = max(float(getattr(port, "timeout", promoted)), promoted)
        except Exception:
            pass


class SharedModbusClient:
    """Reference-counted wrapper around a shared ``ModbusSerialClient``."""

    def __init__(self, key: _SessionKey, client: object):
        self._key = key
        self._client = client
        self._closed = False

    def __getattr__(self, name: str):
        return getattr(self._client, name)

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client_to_close = None
        with _SESSION_LOCK:
            entry = _SESSIONS.get(self._key)
            if entry is None:
                return
            entry.refs -= 1
            if entry.refs <= 0:
                client_to_close = entry.client
                _SESSIONS.pop(self._key, None)
        if client_to_close is not None:
            close = getattr(client_to_close, "close", None)
            if callable(close):
                close()


def open_shared_modbus_client(
    *,
    port: str,
    baudrate: int,
    parity: str,
    stopbits: int,
    bytesize: int,
    timeout_s: float,
) -> SharedModbusClient:
    """Open or reuse a shared ``ModbusSerialClient`` for a serial port."""
    key = _SessionKey(
        port=str(port),
        baudrate=int(baudrate),
        parity=str(parity),
        stopbits=int(stopbits),
        bytesize=int(bytesize),
    )

    with _SESSION_LOCK:
        existing = _SESSIONS.get(key)
        if existing is not None:
            existing.refs += 1
            _apply_timeout(existing.client, timeout_s)
            existing.timeout_s = max(existing.timeout_s, timeout_s)
            return SharedModbusClient(key, existing.client)

    from pymodbus.client import ModbusSerialClient  # type: ignore

    client = ModbusSerialClient(
        port=key.port,
        baudrate=key.baudrate,
        parity=key.parity,
        stopbits=key.stopbits,
        bytesize=key.bytesize,
        timeout=timeout_s,
    )
    if not client.connect():
        raise RuntimeError(f"Modbus connect failed on {key.port}")

    with _SESSION_LOCK:
        existing = _SESSIONS.get(key)
        if existing is not None:
            existing.refs += 1
            _apply_timeout(existing.client, timeout_s)
            existing.timeout_s = max(existing.timeout_s, timeout_s)
            close = getattr(client, "close", None)
            if callable(close):
                close()
            return SharedModbusClient(key, existing.client)

        _SESSIONS[key] = _SessionEntry(client=client, refs=1, timeout_s=timeout_s)
        return SharedModbusClient(key, client)
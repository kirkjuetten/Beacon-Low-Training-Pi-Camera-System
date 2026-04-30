"""Indicator bus abstraction.

The :class:`IndicatorBus` protocol matches the surface used by existing
``IndicatorLED`` callers (``pulse_pass``, ``pulse_fail``, ``cleanup``) and adds
forward-looking primitives (``set_outputs``, ``read_inputs``, ``health_check``)
that the Modbus implementation will use as the system grows beyond two LEDs.

Implementations
---------------
* :class:`NullIndicatorBus` -- safe default for hosts, CI, and units with no
  configured indicator hardware.
* :class:`ModbusIndicatorBus` -- drives the Waveshare USB->RS-485 / Modbus RTU
  8CH IO module (and optionally a Modbus RTU 4CH relay on a separate slave id).
  Hardware is talked to with ``pymodbus`` over a serial port.

The factory :func:`build_indicator_bus` selects the implementation from a
config dict's ``io`` block. When ``io`` is missing or ``io.mode == "gpio"``,
the legacy ``IndicatorLED`` is used so existing deployments are unaffected.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class IndicatorBus(Protocol):
    """Minimal interface used across the inspection system."""

    enabled: bool

    def pulse_pass(self) -> None: ...
    def pulse_fail(self) -> None: ...
    def cleanup(self) -> None: ...


@dataclass
class NullIndicatorBus:
    """No-op bus used when no indicator hardware is configured.

    Records calls in :attr:`history` so the operator self-test can still
    verify the call path end-to-end without real I/O.
    """

    enabled: bool = False
    history: list[str] = field(default_factory=list)

    def pulse_pass(self) -> None:
        self.history.append("pass")

    def pulse_fail(self) -> None:
        self.history.append("fail")

    def cleanup(self) -> None:
        self.history.append("cleanup")


@dataclass
class _ModbusRelayMap:
    """Waveshare Modbus RTU 4CH relay (V3 firmware) configuration.

    When :attr:`pass_channel` and :attr:`fail_channel` are set and the bus's
    ``indicator_target`` is ``"relay"``, ``pulse_pass``/``pulse_fail`` route
    to this module instead of the IO module's coils. The V3 protocol's
    "flash-on" command makes the relay self-time the pulse so the Python
    side stays non-blocking.
    """

    slave_id: int = 2
    pass_channel: Optional[int] = None
    fail_channel: Optional[int] = None


# Waveshare Modbus RTU Relay V3 protocol constants.
_FLASH_ON_BASE_REGISTER = 0x0200  # +channel 0..7
_MAX_FLASH_DECISECONDS = 0x7FFF   # ~54 minutes; protocol limit
_MIN_FLASH_DECISECONDS = 0x0001   # 100 ms minimum (zero is "do nothing")


def _crc16_modbus(data: bytes) -> bytes:
    """Compute the Modbus RTU CRC-16 (low byte first) for ``data``."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def _build_flash_on_frame(slave_id: int, channel: int, deciseconds: int) -> bytes:
    """Build a Waveshare V3 flash-on frame.

    Frame layout: ``[slave] 05 [reg_hi reg_lo] [val_hi val_lo] [crc_lo crc_hi]``
    where ``reg = 0x0200 + channel`` and ``val = duration in 100ms units``.
    """
    if not 0 <= channel <= 7:
        raise ValueError(f"channel must be in 0..7, got {channel}")
    if not _MIN_FLASH_DECISECONDS <= deciseconds <= _MAX_FLASH_DECISECONDS:
        raise ValueError(
            f"deciseconds must be in {_MIN_FLASH_DECISECONDS}..{_MAX_FLASH_DECISECONDS}, "
            f"got {deciseconds}"
        )
    register = _FLASH_ON_BASE_REGISTER + channel
    body = bytes([slave_id & 0xFF, 0x05]) + bytes([
        (register >> 8) & 0xFF, register & 0xFF,
        (deciseconds >> 8) & 0xFF, deciseconds & 0xFF,
    ])
    return body + _crc16_modbus(body)


@dataclass
class ModbusIndicatorBus:
    """Drive a Waveshare Modbus RTU 8CH IO module and/or 4CH relay as indicators.

    Two routing modes via :attr:`indicator_target`:

    * ``"io_module"`` (default, legacy): pulses the IO module's digital outputs
      using ``write_coil`` with a Python ``time.sleep`` for the pulse width.
      Blocking; OK for short pulses (<= 1 s).
    * ``"relay"``: pulses the V3 relay using the protocol's "flash-on" command
      (function 0x05 to register ``0x0200 + channel`` with duration in 100ms
      units). The relay self-times the pulse, so the call returns immediately.
      Required for any pulse longer than ~1 s in production.

    Per-pulse durations :attr:`pass_pulse_ms` and :attr:`fail_pulse_ms` may
    differ; both fall back to :attr:`pulse_ms` when unset (legacy behavior).

    ``pymodbus`` is imported lazily so the rest of the codebase (and CI) does
    not require the dependency unless the Modbus mode is actually selected.
    """

    port: str
    baud: int = 9600
    parity: str = "N"
    stopbits: int = 1
    bytesize: int = 8
    slave_id: int = 1
    pass_channel: int = 0
    fail_channel: int = 1
    pulse_ms: int = 750
    pass_pulse_ms: Optional[int] = None
    fail_pulse_ms: Optional[int] = None
    indicator_target: str = "io_module"
    timeout_s: float = 1.0
    enabled: bool = True
    relay: Optional[_ModbusRelayMap] = None

    _client: object = field(default=None, init=False, repr=False)
    _last_error: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        try:
            from pymodbus.client import ModbusSerialClient  # type: ignore
        except ImportError as exc:
            self._last_error = f"pymodbus not installed: {exc}"
            self.enabled = False
            return

        try:
            client = ModbusSerialClient(
                port=self.port,
                baudrate=self.baud,
                parity=self.parity,
                stopbits=self.stopbits,
                bytesize=self.bytesize,
                timeout=self.timeout_s,
            )
            if not client.connect():
                self._last_error = f"Modbus connect failed on {self.port}"
                self.enabled = False
                return
            self._client = client
        except Exception as exc:  # pragma: no cover - hardware-only path
            self._last_error = f"Modbus init error: {exc}"
            self.enabled = False

    # --- protocol surface ---------------------------------------------------

    def pulse_pass(self) -> None:
        ms = self.pass_pulse_ms if self.pass_pulse_ms is not None else self.pulse_ms
        self._dispatch_pulse(is_pass=True, ms=ms)

    def pulse_fail(self) -> None:
        ms = self.fail_pulse_ms if self.fail_pulse_ms is not None else self.pulse_ms
        self._dispatch_pulse(is_pass=False, ms=ms)

    def cleanup(self) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            # Only the IO-module path sets coils that need explicit clearing.
            # Relay flash-on auto-clears, so nothing to do for that path.
            if self.indicator_target == "io_module":
                self._write_coil(self.pass_channel, False)
                self._write_coil(self.fail_channel, False)
        finally:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._client = None

    # --- richer surface for self-test / future use --------------------------

    def set_outputs(self, channels: Sequence[int], value: bool) -> bool:
        """Set or clear a list of output channels. Returns True on success."""
        if not self.enabled or self._client is None:
            return False
        ok = True
        for ch in channels:
            ok = self._write_coil(ch, value) and ok
        return ok

    def health_check(self) -> dict:
        """Return a small dict describing whether the bus is reachable."""
        return {
            "enabled": self.enabled,
            "port": self.port,
            "slave_id": self.slave_id,
            "last_error": self._last_error,
        }

    # --- internals ----------------------------------------------------------

    def _dispatch_pulse(self, *, is_pass: bool, ms: int) -> None:
        if not self.enabled or self._client is None:
            return
        if self.indicator_target == "relay":
            self._pulse_relay(is_pass=is_pass, ms=ms)
        else:
            channel = self.pass_channel if is_pass else self.fail_channel
            self._pulse_io_module(channel=channel, ms=ms)

    def _pulse_io_module(self, *, channel: int, ms: int) -> None:
        # Clear both indicator channels first so we always end up in a known state.
        self._write_coil(self.pass_channel, False)
        self._write_coil(self.fail_channel, False)
        if not self._write_coil(channel, True):
            return
        time.sleep(ms / 1000.0)
        self._write_coil(channel, False)

    def _pulse_relay(self, *, is_pass: bool, ms: int) -> None:
        if self.relay is None:
            self._last_error = "indicator_target=relay but no relay map configured"
            return
        channel = self.relay.pass_channel if is_pass else self.relay.fail_channel
        if channel is None:
            self._last_error = (
                f"indicator_target=relay but {'pass' if is_pass else 'fail'}_channel is None"
            )
            return
        # Clamp duration to the V3 protocol's range. Round to nearest 100 ms,
        # minimum 1 decisecond (100 ms), maximum 0x7FFF.
        deciseconds = max(_MIN_FLASH_DECISECONDS, min(_MAX_FLASH_DECISECONDS, round(ms / 100)))
        try:
            frame = _build_flash_on_frame(self.relay.slave_id, channel, deciseconds)
        except ValueError as exc:
            self._last_error = f"flash-on frame error: {exc}"
            return
        self._send_raw_frame(frame)

    def _send_raw_frame(self, frame: bytes) -> None:
        """Write a raw Modbus RTU frame via pymodbus's underlying pyserial port.

        Fire-and-forget: we don't read the response. Modbus 0x05 echoes the
        request on success; nothing actionable happens if the frame is lost
        (the next inspection will overwrite the indicator state anyway).
        """
        port = self._raw_serial_port()
        if port is None:
            return
        try:
            port.reset_input_buffer()
            port.write(frame)
            port.flush()
        except Exception as exc:  # pragma: no cover - hardware-only path
            self._last_error = f"raw frame write error: {exc}"

    def _raw_serial_port(self):
        """Return the underlying pyserial Serial held by pymodbus, or None.

        pymodbus 3.x exposes the serial port as ``client.socket``. We probe
        common attribute names to stay tolerant of minor version drift.
        """
        client = self._client
        if client is None:
            return None
        for attr in ("socket", "_socket"):
            port = getattr(client, attr, None)
            if port is not None and hasattr(port, "write"):
                return port
        # pymodbus 3.7+ wraps the port behind a transport object.
        transport = getattr(client, "transport", None)
        if transport is not None:
            for attr in ("serial", "socket"):
                port = getattr(transport, attr, None)
                if port is not None and hasattr(port, "write"):
                    return port
        self._last_error = "could not locate underlying serial port on pymodbus client"
        return None

    def _write_coil(self, channel: int, value: bool) -> bool:
        if self._client is None:
            return False
        try:
            # pymodbus 3.x signature: write_coil(address, value, slave=...)
            result = self._client.write_coil(channel, value, slave=self.slave_id)  # type: ignore[attr-defined]
        except TypeError:
            # pymodbus 2.x signature: write_coil(address, value, unit=...)
            result = self._client.write_coil(channel, value, unit=self.slave_id)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - hardware-only path
            self._last_error = f"write_coil({channel}, {value}) error: {exc}"
            return False
        is_error = getattr(result, "isError", None)
        if callable(is_error) and is_error():
            self._last_error = f"write_coil({channel}, {value}) modbus error: {result}"
            return False
        return True


def build_indicator_bus(config: dict) -> IndicatorBus:
    """Construct the indicator bus selected by ``config['io']`` or fall back to GPIO.

    Selection rules:

    * ``io.mode == "modbus"``: use :class:`ModbusIndicatorBus`.
    * ``io.mode == "none"``: use :class:`NullIndicatorBus`.
    * ``io.mode == "gpio"`` or no ``io`` block: keep legacy
      ``IndicatorLED`` behavior (built from the existing ``indicator_led`` key).
    """
    io_cfg = config.get("io") or {}
    mode = (io_cfg.get("mode") or "gpio").strip().lower()

    if mode == "none":
        return NullIndicatorBus()

    if mode == "modbus":
        modbus_cfg = io_cfg.get("modbus") or {}
        relay_cfg = io_cfg.get("relay") or {}
        relay = None
        if relay_cfg:
            relay = _ModbusRelayMap(
                slave_id=int(relay_cfg.get("slave_id", 2)),
                pass_channel=relay_cfg.get("pass_channel"),
                fail_channel=relay_cfg.get("fail_channel"),
            )
        # Per-pulse durations. None means "fall back to io.pulse_ms".
        pass_pulse_ms_raw = io_cfg.get("pass_pulse_ms")
        fail_pulse_ms_raw = io_cfg.get("fail_pulse_ms")
        return ModbusIndicatorBus(
            port=str(modbus_cfg.get("port", "/dev/ttyUSB0")),
            baud=int(modbus_cfg.get("baud", 9600)),
            parity=str(modbus_cfg.get("parity", "N")),
            stopbits=int(modbus_cfg.get("stopbits", 1)),
            bytesize=int(modbus_cfg.get("bytesize", 8)),
            slave_id=int(modbus_cfg.get("slave_id", 1)),
            pass_channel=int(modbus_cfg.get("pass_channel", 0)),
            fail_channel=int(modbus_cfg.get("fail_channel", 1)),
            pulse_ms=int(io_cfg.get("pulse_ms", 750)),
            pass_pulse_ms=int(pass_pulse_ms_raw) if pass_pulse_ms_raw is not None else None,
            fail_pulse_ms=int(fail_pulse_ms_raw) if fail_pulse_ms_raw is not None else None,
            indicator_target=str(io_cfg.get("indicator_target", "io_module")).strip().lower(),
            timeout_s=float(modbus_cfg.get("timeout_s", 1.0)),
            enabled=bool(modbus_cfg.get("enabled", True)),
            relay=relay,
        )

    # Default / legacy: keep GPIO-based IndicatorLED behavior.
    from inspection_system.app.indicator_context import build_indicator_from_config

    return build_indicator_from_config(config)  # type: ignore[return-value]


def _self_test(argv: list[str] | None = None) -> int:
    """CLI helper: pulse pass and fail on the configured bus once each.

    Run with ``python -m inspection_system.app.io.indicator_bus --self-test``.
    Honors the active project's config; safe to run without hardware -- a Null
    bus will just record the call sequence.
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(prog="indicator_bus", description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="pulse pass then fail once")
    parser.add_argument("--config", type=str, default=None, help="path to a camera_config.json")
    args = parser.parse_args(argv)

    if not args.self_test:
        parser.print_help()
        return 0

    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    else:
        from inspection_system.app.camera_interface import load_config

        config = load_config()

    bus = build_indicator_bus(config)
    print(f"indicator_bus self-test: bus={type(bus).__name__} enabled={getattr(bus, 'enabled', '?')}")
    try:
        bus.pulse_pass()
        print("  pass pulse: ok")
        bus.pulse_fail()
        print("  fail pulse: ok")
    finally:
        bus.cleanup()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    import sys

    sys.exit(_self_test(sys.argv[1:]))

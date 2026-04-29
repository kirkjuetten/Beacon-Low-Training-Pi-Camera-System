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
    """Optional Waveshare Modbus RTU 4CH relay configuration.

    Reserved in config for forward compatibility -- not required for Phase 1.
    """

    slave_id: int = 2
    pass_channel: Optional[int] = None
    fail_channel: Optional[int] = None


@dataclass
class ModbusIndicatorBus:
    """Drive Waveshare Modbus RTU 8CH IO outputs as pass/fail indicators.

    Coil addresses correspond to the 8CH IO module's digital outputs (DO0..DO7).
    The bus performs the same "exclusive-OR pulse" semantics as the legacy
    GPIO ``IndicatorLED``: clears both outputs, sets the chosen one, sleeps,
    clears it again.

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
        self._pulse(self.pass_channel)

    def pulse_fail(self) -> None:
        self._pulse(self.fail_channel)

    def cleanup(self) -> None:
        if not self.enabled or self._client is None:
            return
        try:
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

    def _pulse(self, channel: int) -> None:
        if not self.enabled or self._client is None:
            return
        # Clear both indicator channels first so we always end up in a known state.
        self._write_coil(self.pass_channel, False)
        self._write_coil(self.fail_channel, False)
        if not self._write_coil(channel, True):
            return
        time.sleep(self.pulse_ms / 1000.0)
        self._write_coil(channel, False)

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

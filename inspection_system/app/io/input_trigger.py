"""Modbus discrete-input edge trigger.

Polls a Waveshare 8CH IO module's discrete inputs via Modbus RTU and
fires on rising edges (release-to-press transitions). Used to drive
inspection captures from a physical operator switch.

Design notes
------------
* :meth:`poll` is non-blocking and returns the number of rising edges
  observed since the previous call. The caller (production loop /
  self-test) is responsible for cadence; we recommend ~50 Hz so a
  human button press (typically 100-700 ms wide) is sampled multiple
  times.
* Debouncing is done in software: a candidate edge is only reported
  after the input has been continuously high for at least
  :attr:`debounce_ms`. The Waveshare DIs we tested are clean at the
  Modbus polling rate, but cheap switches still bounce and a 30 ms
  debounce is the standard hardening.
* ``pymodbus`` is imported lazily so CI/test environments are unaffected.
* Both pymodbus 2.x (``unit=``) and 3.x (``slave=``) are supported via
  a ``TypeError`` fallback, matching the convention in
  :mod:`inspection_system.app.io.indicator_bus`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from inspection_system.app.io.modbus_session import open_shared_modbus_client


@runtime_checkable
class InputTrigger(Protocol):
    """Surface used by the production loop and self-test."""

    enabled: bool

    def poll(self) -> int: ...
    def cleanup(self) -> None: ...


@dataclass
class NullInputTrigger:
    """No-op trigger used when no trigger hardware is configured."""

    enabled: bool = False
    history: list[str] = field(default_factory=list)

    def poll(self) -> int:
        self.history.append("poll")
        return 0

    def cleanup(self) -> None:
        self.history.append("cleanup")


@dataclass
class ModbusInputTrigger:
    """Rising-edge trigger driven by a Modbus discrete input.

    Reads ``count`` bits starting at ``register`` on ``slave_id`` and
    watches ``channel`` (0-based offset within that read) for press
    events. A press is registered when the bit transitions from low to
    high and stays high for at least ``debounce_ms``.

    The class manages a shared ``pymodbus`` client so it can coexist with
    :class:`ModbusIndicatorBus` on the same serial port without opening the
    device twice.
    """

    port: str
    baud: int = 9600
    parity: str = "N"
    stopbits: int = 1
    bytesize: int = 8
    slave_id: int = 1
    register: int = 0x0000
    channel: int = 0
    count: int = 8
    debounce_ms: int = 30
    timeout_s: float = 0.3
    enabled: bool = True
    # Injected for tests; production callers leave this None.
    client_factory: Optional[Callable[[], object]] = None

    _client: object = field(default=None, init=False, repr=False)
    _last_error: Optional[str] = field(default=None, init=False, repr=False)
    # Edge-detection state.
    _last_stable_high: bool = field(default=False, init=False, repr=False)
    _candidate_high: bool = field(default=False, init=False, repr=False)
    _candidate_since: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not 0 <= self.channel < self.count:
            self._last_error = (
                f"channel {self.channel} out of range for count={self.count}"
            )
            self.enabled = False
            return
        try:
            client = self._open_client()
        except Exception as exc:  # pragma: no cover - hardware-only path
            self._last_error = f"trigger init error: {exc}"
            self.enabled = False
            return
        if client is None:
            self.enabled = False
            return
        self._client = client

    # --- protocol surface ---------------------------------------------------

    def poll(self) -> int:
        """Read the input once and return the number of rising edges (0 or 1).

        We collapse multiple edges in a single poll into one event because
        the Modbus poll rate (typically <= 50 Hz) is far slower than any
        plausible double-press. The caller drives cadence externally.
        """
        if not self.enabled or self._client is None:
            return 0
        bits = self._read_bits()
        if bits is None:
            return 0
        try:
            high = bool(bits[self.channel])
        except IndexError:
            self._last_error = (
                f"poll: channel {self.channel} not present in {len(bits)}-bit read"
            )
            return 0

        now = time.monotonic()
        if high != self._candidate_high:
            # The signal changed; restart the debounce timer.
            self._candidate_high = high
            self._candidate_since = now

        # Promote candidate to "stable" once it has held its level for at
        # least debounce_ms. A rising-edge event is reported once per
        # low->high stable transition. We evaluate after the optional
        # candidate update so debounce_ms == 0 produces an immediate edge.
        edges = 0
        elapsed_ms = (now - self._candidate_since) * 1000.0
        if (
            elapsed_ms >= self.debounce_ms
            and self._candidate_high != self._last_stable_high
        ):
            if self._candidate_high and not self._last_stable_high:
                edges = 1
            self._last_stable_high = self._candidate_high
        return edges

    def cleanup(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - defensive
                pass

    def health_check(self) -> dict:
        return {
            "enabled": self.enabled,
            "port": self.port,
            "slave_id": self.slave_id,
            "channel": self.channel,
            "register": self.register,
            "last_error": self._last_error,
        }

    # --- internals ----------------------------------------------------------

    def _open_client(self) -> object:
        if self.client_factory is not None:
            return self.client_factory()
        try:
            return open_shared_modbus_client(
                port=self.port,
                baudrate=self.baud,
                parity=self.parity,
                stopbits=self.stopbits,
                bytesize=self.bytesize,
                timeout_s=self.timeout_s,
            )
        except ImportError as exc:
            self._last_error = f"pymodbus not installed: {exc}"
            return None  # type: ignore[return-value]
        except Exception as exc:
            self._last_error = str(exc)
            return None  # type: ignore[return-value]

    def _read_bits(self) -> Optional[list[bool]]:
        client = self._client
        if client is None:
            return None
        read = getattr(client, "read_discrete_inputs", None)
        if not callable(read):
            self._last_error = "client has no read_discrete_inputs"
            return None
        try:
            try:
                response = read(self.register, count=self.count, slave=self.slave_id)
            except TypeError:
                response = read(self.register, count=self.count, unit=self.slave_id)
        except Exception as exc:  # pragma: no cover - hardware-only path
            self._last_error = f"read_discrete_inputs error: {exc}"
            return None
        is_error = getattr(response, "isError", None)
        if callable(is_error) and is_error():
            self._last_error = f"modbus error: {response}"
            return None
        bits = getattr(response, "bits", None)
        if bits is None:
            self._last_error = "modbus response has no bits"
            return None
        return [bool(b) for b in list(bits)[: self.count]]


def build_input_trigger(config: dict) -> InputTrigger:
    """Construct the input trigger selected by ``config['io']['trigger']``.

    When ``io.trigger.enabled`` is False or the ``trigger`` block is
    missing, returns a :class:`NullInputTrigger` so callers can poll
    unconditionally.
    """
    io_cfg = config.get("io") or {}
    trigger_cfg = io_cfg.get("trigger") or {}
    if not trigger_cfg.get("enabled", False):
        return NullInputTrigger()
    modbus_cfg = io_cfg.get("modbus") or {}
    return ModbusInputTrigger(
        port=str(trigger_cfg.get("port", modbus_cfg.get("port", "/dev/ttyUSB0"))),
        baud=int(trigger_cfg.get("baud", modbus_cfg.get("baud", 9600))),
        parity=str(trigger_cfg.get("parity", modbus_cfg.get("parity", "N"))),
        stopbits=int(trigger_cfg.get("stopbits", modbus_cfg.get("stopbits", 1))),
        bytesize=int(trigger_cfg.get("bytesize", modbus_cfg.get("bytesize", 8))),
        slave_id=int(trigger_cfg.get("slave_id", 1)),
        register=int(trigger_cfg.get("register", 0x0000)),
        channel=int(trigger_cfg.get("channel", 0)),
        count=int(trigger_cfg.get("count", 8)),
        debounce_ms=int(trigger_cfg.get("debounce_ms", 30)),
        timeout_s=float(trigger_cfg.get("timeout_s", 0.3)),
    )


def _watch(argv: list[str] | None = None) -> int:
    """CLI helper: poll the configured trigger and print rising-edge events.

    Run with::

        python -m inspection_system.app.io.input_trigger --watch 10 \\
            --config /path/to/camera_config.json

    The --watch argument is the duration in seconds. Useful for bench
    bring-up: wire your switch, run --watch 20, and confirm presses
    show up as edges.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="input_trigger", description=__doc__)
    parser.add_argument(
        "--watch", type=float, default=10.0, help="seconds to poll (default 10)"
    )
    parser.add_argument(
        "--rate", type=float, default=50.0, help="poll rate Hz (default 50)"
    )
    parser.add_argument("--config", type=str, default=None, help="path to camera_config.json")
    args = parser.parse_args(argv)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    else:
        from inspection_system.app.camera_interface import load_config

        config = load_config()

    trigger = build_input_trigger(config)
    print(
        f"input_trigger watch: trigger={type(trigger).__name__} "
        f"enabled={getattr(trigger, 'enabled', '?')}"
    )
    if not getattr(trigger, "enabled", False):
        last_err = getattr(trigger, "_last_error", None)
        if last_err:
            print(f"  trigger disabled: {last_err}")
        return 1

    print(f"Watching for {args.watch:.0f}s -- press your switch now!", flush=True)
    period = 1.0 / max(args.rate, 1.0)
    deadline = time.monotonic() + args.watch
    n_edges = 0
    n_polls = 0
    try:
        while time.monotonic() < deadline:
            edges = trigger.poll()
            n_polls += 1
            if edges:
                n_edges += edges
                t = args.watch - (deadline - time.monotonic())
                print(f"  edge #{n_edges} at t+{t:.3f}s", flush=True)
            time.sleep(period)
    finally:
        trigger.cleanup()
    print(f"watch done: {n_polls} polls, {n_edges} rising edge(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    import sys

    sys.exit(_watch(sys.argv[1:]))

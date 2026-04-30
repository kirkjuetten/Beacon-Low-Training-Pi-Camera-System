"""One-time helper: re-address a Waveshare Modbus RTU relay module.

The Waveshare 4CH (and 8CH) relay modules ship with default Modbus slave
address ``0x01`` -- the same default as the Waveshare 8CH IO module. If both
sit on the same RS-485 bus simultaneously, they will both answer to writes
addressed to slave 1 and the bus will not work reliably.

This tool sends the V3 protocol "set device address" command (write to
holding register ``0x4000``) and then reads the value back to verify the
change took effect.

Usage (run with ONLY the relay module powered on the RS-485 bus)::

    python -m inspection_system.app.io.provision_relay \\
        --port /dev/ttyUSB0 --new-address 2 --confirm

Without ``--confirm``, the tool prints what it would do and exits without
writing. ``--current-address`` defaults to ``1`` (factory default) but can be
overridden if you have already partially provisioned the unit. After a
successful run, power down everything, then re-enable the IO module so the
two devices live on slaves 1 and 2 respectively.

This module imports ``pymodbus`` lazily so the rest of the codebase (and
CI) does not require the dependency unless this tool is actually used.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional, Protocol


_DEVICE_ADDRESS_REGISTER = 0x4000


class _ModbusClient(Protocol):
    """Minimal surface of pymodbus.ModbusSerialClient that we use.

    Declared so tests can pass a fake without importing pymodbus.
    """

    def connect(self) -> bool: ...
    def close(self) -> None: ...
    def read_holding_registers(self, address: int, count: int = 1, **kwargs): ...
    def write_register(self, address: int, value: int, **kwargs): ...


@dataclass
class ProvisionResult:
    """Outcome of a provisioning attempt."""

    success: bool
    message: str
    before_address: Optional[int] = None
    after_address: Optional[int] = None


def _read_address(client: _ModbusClient, slave_id: int) -> Optional[int]:
    """Read register 0x4000 from ``slave_id`` and return its value, or None on error."""
    try:
        # pymodbus 3.x signature: ..., slave=N
        result = client.read_holding_registers(_DEVICE_ADDRESS_REGISTER, count=1, slave=slave_id)
    except TypeError:
        # pymodbus 2.x signature: ..., unit=N
        result = client.read_holding_registers(_DEVICE_ADDRESS_REGISTER, count=1, unit=slave_id)  # type: ignore[call-arg]
    except Exception:
        return None

    is_error = getattr(result, "isError", None)
    if callable(is_error) and is_error():
        return None
    registers = getattr(result, "registers", None)
    if not registers:
        return None
    return int(registers[0])


def _write_address(client: _ModbusClient, slave_id: int, new_address: int) -> bool:
    """Write ``new_address`` to register 0x4000 on ``slave_id``."""
    try:
        result = client.write_register(_DEVICE_ADDRESS_REGISTER, new_address, slave=slave_id)
    except TypeError:
        result = client.write_register(_DEVICE_ADDRESS_REGISTER, new_address, unit=slave_id)  # type: ignore[call-arg]
    except Exception:
        return False

    is_error = getattr(result, "isError", None)
    if callable(is_error) and is_error():
        return False
    return True


def provision(
    client: _ModbusClient,
    *,
    current_address: int,
    new_address: int,
    settle_seconds: float = 0.3,
) -> ProvisionResult:
    """Re-address a Modbus device using a pre-connected ``client``.

    Sequence:

    1. Read register 0x4000 from ``current_address`` to sanity-check the device
       is present and reports the expected current address.
    2. Write ``new_address`` to register 0x4000.
    3. Sleep briefly to let the device latch the new address.
    4. Read register 0x4000 from ``new_address`` and confirm it matches.
    """
    if not (1 <= new_address <= 0xFF):
        return ProvisionResult(False, f"new-address {new_address} out of range 1..255")
    if new_address == current_address:
        return ProvisionResult(
            False,
            f"new-address {new_address} equals current-address; nothing to do",
        )

    before = _read_address(client, current_address)
    if before is None:
        return ProvisionResult(
            False,
            f"no response from slave {current_address}; "
            "check that the relay is powered, A/B wiring is correct, "
            "and no other device on the bus shares this address",
        )
    if before != current_address:
        return ProvisionResult(
            False,
            f"slave {current_address} reports its address as {before}; aborting to be safe",
            before_address=before,
        )

    if not _write_address(client, current_address, new_address):
        return ProvisionResult(
            False,
            f"write to register 0x{_DEVICE_ADDRESS_REGISTER:04X} failed",
            before_address=before,
        )

    time.sleep(settle_seconds)

    after = _read_address(client, new_address)
    if after is None:
        return ProvisionResult(
            False,
            f"no response at new address {new_address} after write",
            before_address=before,
        )
    if after != new_address:
        return ProvisionResult(
            False,
            f"verification mismatch: device at {new_address} reports {after}",
            before_address=before,
            after_address=after,
        )

    return ProvisionResult(
        True,
        f"address changed: {before} -> {after}",
        before_address=before,
        after_address=after,
    )


def _open_serial_client(port: str, baud: int, parity: str, timeout_s: float):
    """Construct and connect a pymodbus serial client. Returns the client or None."""
    try:
        from pymodbus.client import ModbusSerialClient  # type: ignore
    except ImportError as exc:
        print(f"pymodbus not installed: {exc}", file=sys.stderr)
        return None

    client = ModbusSerialClient(
        port=port,
        baudrate=baud,
        parity=parity,
        stopbits=1,
        bytesize=8,
        timeout=timeout_s,
    )
    if not client.connect():
        print(f"could not open serial port {port}", file=sys.stderr)
        return None
    return client


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="provision_relay",
        description="Re-address a Waveshare Modbus RTU relay (V3 protocol).",
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--parity", default="N", choices=["N", "E", "O"])
    parser.add_argument("--timeout", type=float, default=1.0, help="per-request timeout seconds")
    parser.add_argument(
        "--current-address",
        type=int,
        default=1,
        help="address the relay currently answers to (default: 1, the factory default)",
    )
    parser.add_argument(
        "--new-address",
        type=int,
        default=2,
        help="address to assign (default: 2)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="actually perform the write; without this flag the tool runs in dry-run mode",
    )
    args = parser.parse_args(argv)

    print("Waveshare relay address provisioning")
    print(f"  port           : {args.port}")
    print(f"  current address: {args.current_address}")
    print(f"  new address    : {args.new_address}")
    print()
    print("Reminder: ONLY the relay module should be powered on the RS-485 bus")
    print("right now. If the IO module is also on the bus, address conflicts")
    print("will make this command unreliable.")
    print()

    if not args.confirm:
        print("Dry run -- not writing. Re-run with --confirm to apply.")
        return 0

    client = _open_serial_client(args.port, args.baud, args.parity, args.timeout)
    if client is None:
        return 1

    try:
        result = provision(
            client,
            current_address=args.current_address,
            new_address=args.new_address,
        )
    finally:
        try:
            client.close()
        except Exception:
            pass

    if result.success:
        print(f"OK: {result.message}")
        print()
        print("Power the relay off and on once to make sure the new address")
        print("survives a reset, then re-enable the IO module on the bus.")
        return 0

    print(f"FAIL: {result.message}", file=sys.stderr)
    if result.before_address is not None:
        print(f"  observed before-address: {result.before_address}", file=sys.stderr)
    if result.after_address is not None:
        print(f"  observed after-address : {result.after_address}", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main(sys.argv[1:]))

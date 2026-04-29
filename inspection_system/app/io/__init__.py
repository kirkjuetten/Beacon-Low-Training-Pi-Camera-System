"""I/O abstractions for the inspection system.

See :mod:`inspection_system.app.io.indicator_bus` for the protocol and
implementations covering the legacy GPIO indicator and the Waveshare
USB->RS-485 / Modbus RTU 8CH IO module (with optional 4CH relay support).

Submodules are imported on demand to keep the package lightweight and to
avoid runpy warnings when running submodules with ``python -m``.
"""

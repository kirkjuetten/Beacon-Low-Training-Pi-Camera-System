#!/usr/bin/env python3
"""Shared subprocess launch helpers for UI surfaces."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path


def stream_command(
    command: Sequence[str],
    *,
    cwd: str | Path,
    on_output: Callable[[str], None],
    on_complete: Callable[[int], None],
    on_error: Callable[[OSError], None] | None = None,
) -> threading.Thread:
    """Run a command on a worker thread and stream output lines to callbacks."""

    def run() -> None:
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            if on_error is not None:
                on_error(exc)
            return

        assert process.stdout is not None
        for line in process.stdout:
            on_output(line)

        on_complete(process.wait())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def launch_monitored_command(
    command: Sequence[str],
    *,
    cwd: str | Path,
    on_output: Callable[[str], None],
    on_exit: Callable[[int], None],
) -> subprocess.Popen[str]:
    """Launch a command immediately and monitor its output on a background thread."""

    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def monitor_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            on_output(line)
        on_exit(process.wait())

    threading.Thread(target=monitor_output, daemon=True).start()
    return process
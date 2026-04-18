#!/usr/bin/env python3
"""Touch keyboard helpers for Tk-based operator UIs."""

from __future__ import annotations

import shutil
import subprocess
from typing import Callable, Optional


def select_touch_keyboard_command(which_fn: Callable[[str], Optional[str]] = shutil.which) -> list[str] | None:
    """Pick the best available on-screen keyboard command for Linux Pi setups."""
    candidates = [
        ["wvkbd-mobintl"],
        ["wvkbd"],
        ["matchbox-keyboard"],
        ["onboard"],
    ]

    for cmd in candidates:
        if which_fn(cmd[0]):
            return cmd
    return None


class TouchKeyboardManager:
    """Shows an on-screen keyboard when text inputs gain focus."""

    def __init__(self, root, *, hide_delay_ms: int = 180):
        self.root = root
        self.hide_delay_ms = hide_delay_ms
        self.launch_cmd = select_touch_keyboard_command()
        self._process: subprocess.Popen | None = None
        self._hide_timer: str | None = None

        if self.launch_cmd:
            self._bind_input_focus_events()

    @property
    def enabled(self) -> bool:
        return self.launch_cmd is not None

    def _bind_input_focus_events(self) -> None:
        # Bind both classic Tk and ttk widget classes used for text entry.
        for klass in ("Entry", "TEntry", "Spinbox", "TCombobox"):
            self.root.bind_class(klass, "<FocusIn>", self._on_focus_in, add="+")
            self.root.bind_class(klass, "<FocusOut>", self._on_focus_out, add="+")

    def _on_focus_in(self, _event=None) -> None:
        if not self.enabled:
            return
        if self._hide_timer is not None:
            try:
                self.root.after_cancel(self._hide_timer)
            except Exception:
                pass
            self._hide_timer = None

        if self._process is None or self._process.poll() is not None:
            try:
                self._process = subprocess.Popen(self.launch_cmd)
            except OSError:
                self._process = None

    def _on_focus_out(self, _event=None) -> None:
        if not self.enabled:
            return
        if self._hide_timer is not None:
            try:
                self.root.after_cancel(self._hide_timer)
            except Exception:
                pass

        self._hide_timer = self.root.after(self.hide_delay_ms, self.hide_keyboard)

    def hide_keyboard(self) -> None:
        self._hide_timer = None
        if self._process is None:
            return

        if self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=0.8)
            except subprocess.TimeoutExpired:
                try:
                    self._process.kill()
                except OSError:
                    pass
            except OSError:
                pass
        self._process = None

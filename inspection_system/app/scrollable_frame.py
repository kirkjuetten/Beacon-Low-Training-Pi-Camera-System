#!/usr/bin/env python3
"""Reusable vertical scrolling container for Tk-based operator screens."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class VerticalScrolledFrame(ttk.Frame):
    """Frame with an always-visible vertical scrollbar and width-synced content."""

    def __init__(self, parent, *, content_padding=0, canvas_background: str | None = None):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        canvas_kwargs: dict[str, object] = {"highlightthickness": 0}
        if canvas_background is not None:
            canvas_kwargs["bg"] = canvas_background

        self.canvas = tk.Canvas(self, **canvas_kwargs)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.content = ttk.Frame(self.canvas, padding=content_padding)
        self.content.columnconfigure(0, weight=1)
        self._window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._sync_content_width)

    def _sync_scroll_region(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_content_width(self, event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)

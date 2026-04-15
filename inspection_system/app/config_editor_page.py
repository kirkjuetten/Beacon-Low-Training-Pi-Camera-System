#!/usr/bin/env python3
"""Dedicated config editor with side-by-side preview for tuning workflows."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from inspection_system.app.camera_interface import get_active_runtime_paths, get_current_project
from inspection_system.app.frame_acquisition import capture_to_temp, cleanup_temp_image
from inspection_system.app.operator_dashboard import (
    CAPTURE_SCRIPT,
    PIL_AVAILABLE,
    REFERENCE_PREVIEW_NAME,
    LIVE_PREVIEW_NAME,
    CONFIG_FIELD_SPECS,
    apply_config_updates,
    build_config_editor_values,
    describe_preview_image,
    find_preview_image,
    read_json_file,
    write_json_file,
)

if PIL_AVAILABLE:
    from PIL import Image, ImageTk
else:
    Image = None
    ImageTk = None


class ConfigEditorPage:
    """Full-screen config tuning view with integrated preview panel."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Beacon Config + Preview")
        self.root.attributes("-fullscreen", True)

        self.config_vars: dict[str, tk.StringVar] = {}
        self.preview_photo = None
        self.current_preview_path: str | None = None
        self._preview_render_job: str | None = None
        self.live_capture_path: Path | None = None
        self.display_mode = "stored"
        self.busy = False

        self.status_var = tk.StringVar(value="Ready")
        self.current_project_var = tk.StringVar(value="Current project: None")
        self.active_config_var = tk.StringVar(value="Config: -")
        self.active_reference_var = tk.StringVar(value="Reference: -")
        self.active_log_var = tk.StringVar(value="Logs: -")
        self.preview_path_var = tk.StringVar(value="Preview: none")

        self._build_layout()
        self.refresh_view()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=5, uniform="main_panels")
        main.columnconfigure(1, weight=6, uniform="main_panels")
        main.rowconfigure(1, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Config + Preview", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_var).grid(row=0, column=1, sticky="e")

        meta = ttk.Frame(main)
        meta.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        meta.columnconfigure(0, weight=1)
        meta.rowconfigure(1, weight=1)

        info = ttk.LabelFrame(meta, text="Active Runtime", padding=10)
        info.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        info.columnconfigure(0, weight=1)
        ttk.Label(info, textvariable=self.current_project_var).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(info, textvariable=self.active_config_var, wraplength=480).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(info, textvariable=self.active_reference_var, wraplength=480).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(info, textvariable=self.active_log_var, wraplength=480).grid(row=3, column=0, sticky="w", pady=2)

        preview = ttk.LabelFrame(meta, text="Latest Preview", padding=10)
        preview.grid(row=1, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)

        ttk.Label(preview, textvariable=self.preview_path_var, wraplength=480).grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(preview, text="No preview image available", anchor="center")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.preview_label.bind("<Configure>", self._schedule_preview_render)

        preview_buttons = ttk.Frame(preview)
        preview_buttons.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        preview_buttons.columnconfigure(0, weight=1)
        preview_buttons.columnconfigure(1, weight=1)
        preview_buttons.columnconfigure(2, weight=1)
        self.capture_button = ttk.Button(preview_buttons, text="Capture", command=self.capture_live_preview)
        self.capture_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.save_preview_button = ttk.Button(preview_buttons, text="Save", command=self.save_live_preview)
        self.save_preview_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.stored_button = ttk.Button(preview_buttons, text="Stored", command=self.show_stored_preview)
        self.stored_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        config = ttk.LabelFrame(main, text="Config Editor", padding=10)
        config.grid(row=1, column=1, sticky="nsew")
        config.columnconfigure(0, weight=1)
        config.rowconfigure(0, weight=1)

        canvas = tk.Canvas(config, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(config, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        form = ttk.Frame(canvas)
        form.columnconfigure(1, weight=1)
        window_id = canvas.create_window((0, 0), window=form, anchor="nw")

        def _sync_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(_event) -> None:
            canvas.itemconfigure(window_id, width=_event.width)

        form.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", _sync_width)

        for row, (dotted_path, label, _) in enumerate(CONFIG_FIELD_SPECS):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))
            var = tk.StringVar()
            self.config_vars[dotted_path] = var
            ttk.Entry(form, textvariable=var).grid(row=row, column=1, sticky="ew", pady=3)

        actions = ttk.Frame(config)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        for idx in range(4):
            actions.columnconfigure(idx, weight=1)

        self.reload_button = ttk.Button(actions, text="Reload Config", command=self.reload_config_editor)
        self.reload_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.save_button = ttk.Button(actions, text="Save Config", command=self.save_config_editor)
        self.save_button.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        self.back_button = ttk.Button(actions, text="Back to Dashboard", command=self.back_to_dashboard)
        self.back_button.grid(row=0, column=2, sticky="ew", padx=(4, 4))
        self.exit_button = ttk.Button(actions, text="Exit", command=self.exit_page)
        self.exit_button.grid(row=0, column=3, sticky="ew", padx=(4, 0))

    def _schedule_preview_render(self, _event=None) -> None:
        if self._preview_render_job is not None:
            try:
                self.root.after_cancel(self._preview_render_job)
            except Exception:
                pass
        self._preview_render_job = self.root.after(50, self._render_preview_image)

    def _render_preview_image(self) -> None:
        self._preview_render_job = None
        if self.current_preview_path is None or not PIL_AVAILABLE:
            return

        label_width = max(1, int(self.preview_label.winfo_width()))
        label_height = max(1, int(self.preview_label.winfo_height()))
        if label_width <= 1 or label_height <= 1:
            self._schedule_preview_render()
            return

        image = Image.open(self.current_preview_path)
        image.thumbnail((max(1, label_width - 8), max(1, label_height - 8)))
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo, text="")

    def set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.status_var.set(status)
        state = "disabled" if busy else "normal"
        for button in [
            self.capture_button,
            self.reload_button,
            self.back_button,
            self.exit_button,
        ]:
            button.configure(state=state)
        self.reload_button.configure(state=state)
        self.save_button.configure(state=state)
        self._update_preview_button_states()

    def _update_preview_button_states(self) -> None:
        if self.busy:
            self.save_preview_button.configure(state="disabled")
            self.stored_button.configure(state="disabled")
            return
        self.save_preview_button.configure(state="normal" if self.live_capture_path is not None else "disabled")
        self.stored_button.configure(state="normal")

    def _clear_live_capture(self) -> None:
        if self.live_capture_path is not None:
            try:
                if self.live_capture_path.exists():
                    self.live_capture_path.unlink()
            except OSError:
                pass
        self.live_capture_path = None

    def _show_preview_image(self, preview_path: Path, label_text: str, *, mode: str) -> None:
        self.display_mode = mode
        self.current_preview_path = str(preview_path)
        self.preview_path_var.set(label_text)
        if PIL_AVAILABLE:
            self._schedule_preview_render()
        else:
            self.preview_label.configure(text=str(preview_path), image="")
            self.preview_photo = None

    def _show_no_preview(self, message: str) -> None:
        self.display_mode = "stored"
        self.current_preview_path = None
        self.preview_path_var.set(message)
        self.preview_label.configure(text="No preview image available", image="")
        self.preview_photo = None

    def _get_stored_preview_path(self) -> Path | None:
        return find_preview_image(get_active_runtime_paths()["reference_dir"])

    def back_to_dashboard(self) -> None:
        try:
            self._clear_live_capture()
            subprocess.Popen([sys.executable, str(CAPTURE_SCRIPT), "dashboard"])
            self.root.destroy()
        except OSError as exc:
            messagebox.showerror("Launch failed", f"Could not launch dashboard: {exc}")

    def exit_page(self) -> None:
        self._clear_live_capture()
        self.root.destroy()

    def reload_config_editor(self) -> None:
        config = read_json_file(get_active_runtime_paths()["config_file"])
        for dotted_path, value in build_config_editor_values(config).items():
            self.config_vars[dotted_path].set(value)

    def save_config_editor(self) -> None:
        active_paths = get_active_runtime_paths()
        config_path = active_paths["config_file"]
        try:
            config = read_json_file(config_path)
            updated = apply_config_updates(config, {key: var.get() for key, var in self.config_vars.items()})
            write_json_file(config_path, updated)
            self.status_var.set(f"Saved config updates to {config_path}")
            self.refresh_view()
        except ValueError as exc:
            messagebox.showerror("Invalid config value", str(exc))

    def capture_live_preview(self) -> None:
        if self.busy:
            return
        self.set_busy(True, "Capturing live preview...")
        thread = threading.Thread(target=self._capture_live_preview_thread, daemon=True)
        thread.start()

    def _capture_live_preview_thread(self) -> None:
        active_paths = get_active_runtime_paths()
        config = read_json_file(active_paths["config_file"])
        result_code = 1
        stderr_text = ""
        live_capture_copy: Path | None = None
        try:
            result_code, image_path, stderr_text = capture_to_temp(config)
            if result_code == 0 and image_path.exists():
                fd, temp_path_text = tempfile.mkstemp(prefix="beacon-live-preview-", suffix=image_path.suffix)
                live_capture_copy = Path(temp_path_text)
                try:
                    Path(temp_path_text).unlink(missing_ok=True)
                except TypeError:
                    if Path(temp_path_text).exists():
                        Path(temp_path_text).unlink()
                finally:
                    try:
                        import os
                        os.close(fd)
                    except OSError:
                        pass
                shutil.copy2(image_path, live_capture_copy)
            else:
                stderr_text = stderr_text or "capture failed"
        finally:
            cleanup_temp_image()

        def finish() -> None:
            if result_code == 0:
                self._clear_live_capture()
                self.live_capture_path = live_capture_copy
                assert live_capture_copy is not None
                self._show_preview_image(live_capture_copy, f"Preview (live): {live_capture_copy.name}", mode="live")
                self.set_busy(False, "Live preview captured")
            else:
                self.set_busy(False, f"Live preview failed: {stderr_text}")
                self.refresh_view()

        self.root.after(0, finish)

    def save_live_preview(self) -> None:
        if self.busy or self.live_capture_path is None:
            return
        active_paths = get_active_runtime_paths()
        preview_path = active_paths["reference_dir"] / LIVE_PREVIEW_NAME
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.live_capture_path, preview_path)
        self._clear_live_capture()
        self.set_busy(False, f"Saved preview to {preview_path.name}")
        self.show_stored_preview()

    def show_stored_preview(self) -> None:
        self.display_mode = "stored"
        self.refresh_view()

    def refresh_preview(self) -> None:
        if self.display_mode == "live" and self.live_capture_path is not None and self.live_capture_path.exists():
            self._show_preview_image(self.live_capture_path, f"Preview (live): {self.live_capture_path.name}", mode="live")
            self._update_preview_button_states()
            return

        preview_path = self._get_stored_preview_path()
        if preview_path is None:
            self._show_no_preview("Preview: none")
            self._update_preview_button_states()
            return

        preview_kind = describe_preview_image(preview_path)
        self._show_preview_image(preview_path, f"Preview (stored {preview_kind}): {preview_path.name}", mode="stored")
        self._update_preview_button_states()

    def refresh_view(self) -> None:
        current_project = get_current_project() or "None"
        self.current_project_var.set(f"Current project: {current_project}")

        active_paths = get_active_runtime_paths()
        self.active_config_var.set(f"Config: {active_paths['config_file']}")
        self.active_reference_var.set(f"Reference: {active_paths['reference_dir']}")
        self.active_log_var.set(f"Logs: {active_paths['log_dir']}")

        self.reload_config_editor()
        if self.display_mode != "live":
            self.display_mode = "stored"
        self.refresh_preview()


def main() -> None:
    root = tk.Tk()
    ConfigEditorPage(root)
    root.mainloop()


if __name__ == "__main__":
    main()
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
    CONFIG_DROPDOWN_OPTIONS,
    apply_config_updates,
    build_config_editor_values,
    describe_preview_image,
    find_preview_image,
    read_json_file,
    write_json_file,
)
from inspection_system.app.reference_service import set_reference

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_TUNING_DOC = REPO_ROOT / "docs" / "CONFIG_TUNING.md"

if PIL_AVAILABLE:
    from PIL import Image, ImageTk
else:
    Image = None
    ImageTk = None


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ROISetupDialog:
    """Visual ROI setup dialog with image overlay and drag selection."""

    def __init__(self, parent: tk.Tk, on_saved, initial_image_path: Path | None = None):
        self.parent = parent
        self.on_saved = on_saved
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("ROI Setup")
        self.dialog.geometry("1200x780")
        self.dialog.minsize(900, 620)
        self.dialog.transient(parent)

        self.status_var = tk.StringVar(value="Load or capture an image, then drag to draw ROI.")
        self.roi_x_var = tk.StringVar(value="0")
        self.roi_y_var = tk.StringVar(value="0")
        self.roi_w_var = tk.StringVar(value="0")
        self.roi_h_var = tk.StringVar(value="0")

        self.original_image = None
        self.canvas_photo = None
        self.current_image_path: Path | None = None
        self.temp_capture_path: Path | None = None
        self._drag_anchor: tuple[int, int] | None = None
        self._syncing_vars = False

        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._render_w = 1
        self._render_h = 1

        self.roi = {"x": 0, "y": 0, "width": 0, "height": 0}

        self._build_layout()
        self._load_roi_from_config()

        if initial_image_path is not None and initial_image_path.exists():
            self._set_image(initial_image_path)
        else:
            self._load_stored_image()

        self.dialog.protocol("WM_DELETE_WINDOW", self.close)

    def _build_layout(self) -> None:
        self.dialog.columnconfigure(0, weight=4)
        self.dialog.columnconfigure(1, weight=2)
        self.dialog.rowconfigure(0, weight=1)

        canvas_frame = ttk.LabelFrame(self.dialog, text="ROI Image", padding=10)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self._render_canvas())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        controls = ttk.LabelFrame(self.dialog, text="ROI Controls", padding=10)
        controls.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="X").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.roi_x_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(controls, text="Y").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.roi_y_var).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(controls, text="Width").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.roi_w_var).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(controls, text="Height").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.roi_h_var).grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Separator(controls, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)

        ttk.Button(controls, text="Capture", command=self._capture_image).grid(row=5, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(controls, text="Stored", command=self._load_stored_image).grid(row=6, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(controls, text="Full Frame", command=self._set_full_frame_roi).grid(row=7, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(controls, text="Save ROI", command=self._save_roi).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(12, 4))
        ttk.Button(controls, text="Close", command=self.close).grid(row=9, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Label(
            controls,
            text=(
                "Tip: click and drag over the image to define ROI.\n"
                "You can also fine tune X/Y/Width/Height fields."
            ),
            wraplength=280,
            justify="left",
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(12, 4))

        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=280,
            justify="left",
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(8, 0))

        for var in [self.roi_x_var, self.roi_y_var, self.roi_w_var, self.roi_h_var]:
            var.trace_add("write", self._on_roi_vars_changed)

    def _load_roi_from_config(self) -> None:
        config = read_json_file(get_active_runtime_paths()["config_file"])
        roi_cfg = config.get("inspection", {}).get("roi", {})
        self.roi = {
            "x": _safe_int(roi_cfg.get("x", 0), 0),
            "y": _safe_int(roi_cfg.get("y", 0), 0),
            "width": _safe_int(roi_cfg.get("width", 0), 0),
            "height": _safe_int(roi_cfg.get("height", 0), 0),
        }
        self._sync_vars_from_roi()

    def _sync_vars_from_roi(self) -> None:
        self._syncing_vars = True
        try:
            self.roi_x_var.set(str(self.roi["x"]))
            self.roi_y_var.set(str(self.roi["y"]))
            self.roi_w_var.set(str(self.roi["width"]))
            self.roi_h_var.set(str(self.roi["height"]))
        finally:
            self._syncing_vars = False

    def _apply_vars_to_roi(self) -> None:
        x = max(0, _safe_int(self.roi_x_var.get(), self.roi["x"]))
        y = max(0, _safe_int(self.roi_y_var.get(), self.roi["y"]))
        w = max(0, _safe_int(self.roi_w_var.get(), self.roi["width"]))
        h = max(0, _safe_int(self.roi_h_var.get(), self.roi["height"]))

        if self.original_image is not None:
            max_w, max_h = self.original_image.size
            x = min(x, max_w)
            y = min(y, max_h)
            w = min(w, max(0, max_w - x))
            h = min(h, max(0, max_h - y))

        self.roi = {"x": x, "y": y, "width": w, "height": h}

    def _on_roi_vars_changed(self, *_args) -> None:
        if self._syncing_vars:
            return
        self._apply_vars_to_roi()
        self._render_canvas()

    def _set_image(self, image_path: Path) -> None:
        if not PIL_AVAILABLE:
            self.status_var.set("Pillow not installed. Cannot render ROI image preview.")
            return
        try:
            self.original_image = Image.open(image_path).convert("RGB")
            self.current_image_path = image_path
        except Exception as exc:
            self.status_var.set(f"Failed to load image: {exc}")
            return

        if self.roi["width"] <= 0 or self.roi["height"] <= 0:
            self._set_full_frame_roi()
        self._render_canvas()
        self.status_var.set(f"Loaded image: {image_path.name}")

    def _load_stored_image(self) -> None:
        preview_path = find_preview_image(get_active_runtime_paths()["reference_dir"])
        if preview_path is None:
            self.status_var.set("No stored preview image found.")
            return
        self._set_image(preview_path)

    def _capture_image(self) -> None:
        config = read_json_file(get_active_runtime_paths()["config_file"])
        result_code, image_path, stderr_text = capture_to_temp(config)
        try:
            if result_code != 0:
                self.status_var.set(f"Capture failed: {stderr_text or 'unknown error'}")
                return
            fd, temp_path_text = tempfile.mkstemp(prefix="beacon-roi-preview-", suffix=image_path.suffix)
            temp_path = Path(temp_path_text)
            temp_path.unlink(missing_ok=True)
            try:
                import os
                os.close(fd)
            except OSError:
                pass
            shutil.copy2(image_path, temp_path)
            if self.temp_capture_path is not None and self.temp_capture_path.exists():
                self.temp_capture_path.unlink(missing_ok=True)
            self.temp_capture_path = temp_path
            self._set_image(temp_path)
        finally:
            cleanup_temp_image()

    def _set_full_frame_roi(self) -> None:
        if self.original_image is None:
            self.status_var.set("Load an image first to set full-frame ROI.")
            return
        width, height = self.original_image.size
        self.roi = {"x": 0, "y": 0, "width": int(width), "height": int(height)}
        self._sync_vars_from_roi()
        self._render_canvas()

    def _render_canvas(self) -> None:
        self.canvas.delete("all")

        canvas_w = max(1, int(self.canvas.winfo_width()))
        canvas_h = max(1, int(self.canvas.winfo_height()))
        if self.original_image is None:
            self.canvas.create_text(canvas_w // 2, canvas_h // 2, text="No image loaded", fill="#cccccc")
            return

        img_w, img_h = self.original_image.size
        self._scale = min(canvas_w / max(1, img_w), canvas_h / max(1, img_h))
        self._render_w = max(1, int(img_w * self._scale))
        self._render_h = max(1, int(img_h * self._scale))
        self._offset_x = (canvas_w - self._render_w) // 2
        self._offset_y = (canvas_h - self._render_h) // 2

        rendered = self.original_image.resize((self._render_w, self._render_h))
        self.canvas_photo = ImageTk.PhotoImage(rendered)
        self.canvas.create_image(self._offset_x, self._offset_y, image=self.canvas_photo, anchor="nw")

        x1 = self._offset_x + int(self.roi["x"] * self._scale)
        y1 = self._offset_y + int(self.roi["y"] * self._scale)
        x2 = self._offset_x + int((self.roi["x"] + self.roi["width"]) * self._scale)
        y2 = self._offset_y + int((self.roi["y"] + self.roi["height"]) * self._scale)

        if x2 > x1 and y2 > y1:
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff00", width=2)
            self.canvas.create_text(x1 + 6, y1 + 6, anchor="nw", fill="#00ff00", text="ROI")

    def _canvas_to_image(self, canvas_x: int, canvas_y: int) -> tuple[int, int]:
        if self.original_image is None:
            return 0, 0
        img_w, img_h = self.original_image.size
        x = int((canvas_x - self._offset_x) / max(1e-6, self._scale))
        y = int((canvas_y - self._offset_y) / max(1e-6, self._scale))
        x = min(max(0, x), img_w)
        y = min(max(0, y), img_h)
        return x, y

    def _on_canvas_press(self, event) -> None:
        if self.original_image is None:
            return
        start_x, start_y = self._canvas_to_image(int(event.x), int(event.y))
        self._drag_anchor = (start_x, start_y)
        self.roi = {"x": start_x, "y": start_y, "width": 1, "height": 1}
        self._sync_vars_from_roi()
        self._render_canvas()

    def _on_canvas_drag(self, event) -> None:
        if self.original_image is None or self._drag_anchor is None:
            return
        end_x, end_y = self._canvas_to_image(int(event.x), int(event.y))
        start_x, start_y = self._drag_anchor
        x = min(start_x, end_x)
        y = min(start_y, end_y)
        width = abs(end_x - start_x)
        height = abs(end_y - start_y)
        self.roi = {"x": x, "y": y, "width": width, "height": height}
        self._sync_vars_from_roi()
        self._render_canvas()

    def _on_canvas_release(self, _event) -> None:
        self._drag_anchor = None

    def _save_roi(self) -> None:
        self._apply_vars_to_roi()
        if self.roi["width"] <= 0 or self.roi["height"] <= 0:
            messagebox.showerror("Invalid ROI", "ROI width and height must be greater than zero.")
            return

        config_path = get_active_runtime_paths()["config_file"]
        config = read_json_file(config_path)
        inspection = config.setdefault("inspection", {})
        inspection["roi"] = {
            "x": int(self.roi["x"]),
            "y": int(self.roi["y"]),
            "width": int(self.roi["width"]),
            "height": int(self.roi["height"]),
        }
        write_json_file(config_path, config)
        self.status_var.set(f"Saved ROI to {config_path}")
        if self.on_saved is not None:
            self.on_saved()

    def close(self) -> None:
        if self.temp_capture_path is not None and self.temp_capture_path.exists():
            self.temp_capture_path.unlink(missing_ok=True)
        self.dialog.destroy()


class ConfigEditorPage:
    """Full-screen config tuning view with integrated preview panel."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Beacon Config + Preview")
        self.root.attributes("-fullscreen", True)

        self.config_vars: dict[str, tk.StringVar] = {}
        self.config_widgets: dict[str, tk.Widget] = {}
        self.preview_photo = None
        self.current_preview_path: str | None = None
        self._preview_render_job: str | None = None
        self.live_capture_path: Path | None = None
        self.display_mode = "stored"
        self.busy = False

        self.status_var = tk.StringVar(value="Ready")
        self.current_project_var = tk.StringVar(value="Current project: None")
        self.preview_path_var = tk.StringVar(value="Preview: none")
        self.roi_dialog: ROISetupDialog | None = None

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
        ttk.Label(info, textvariable=self.current_project_var).grid(row=0, column=0, sticky="w")

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
        self.stored_button = ttk.Button(preview_buttons, text="Stored", command=self.show_stored_preview)
        self.stored_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.set_reference_button = ttk.Button(preview_buttons, text="Set Reference", command=self.set_reference_from_config)
        self.set_reference_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

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
            if dotted_path in CONFIG_DROPDOWN_OPTIONS:
                widget = ttk.Combobox(
                    form,
                    textvariable=var,
                    values=CONFIG_DROPDOWN_OPTIONS[dotted_path],
                    state="readonly",
                )
            else:
                widget = ttk.Entry(form, textvariable=var)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            self.config_widgets[dotted_path] = widget

        actions = ttk.Frame(config)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        for idx in range(5):
            actions.columnconfigure(idx, weight=1)

        self.reload_button = ttk.Button(actions, text="Reload", command=self.reload_config_editor)
        self.reload_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.save_button = ttk.Button(actions, text="Save", command=self.save_config_editor)
        self.save_button.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        self.back_button = ttk.Button(actions, text="Back", command=self.back_to_dashboard)
        self.back_button.grid(row=0, column=2, sticky="ew", padx=(4, 4))
        self.roi_button = ttk.Button(actions, text="ROI", command=self.open_roi_setup)
        self.roi_button.grid(row=0, column=3, sticky="ew", padx=(4, 4))
        self.info_button = ttk.Button(actions, text="Info", command=self.show_settings_info)
        self.info_button.grid(row=0, column=4, sticky="ew", padx=(4, 0))

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
            self.set_reference_button,
            self.reload_button,
            self.back_button,
            self.roi_button,
            self.info_button,
        ]:
            button.configure(state=state)
        self.reload_button.configure(state=state)
        self.save_button.configure(state=state)
        self._update_preview_button_states()

    def _load_settings_info_text(self) -> str:
        if CONFIG_TUNING_DOC.exists():
            return CONFIG_TUNING_DOC.read_text(encoding="utf-8")
        return (
            "Config tuning guide not found.\n\n"
            "Expected file:\n"
            f"{CONFIG_TUNING_DOC}\n"
        )

    def show_settings_info(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Config Settings Info")
        dialog.transient(self.root)
        dialog.geometry("980x700")
        dialog.minsize(760, 520)

        frame = ttk.Frame(dialog, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Config Tuning Guide", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))

        text_widget = tk.Text(frame, wrap="word", font=("Segoe UI", 10), padx=10, pady=10)
        text_widget.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.insert("1.0", self._load_settings_info_text())
        text_widget.configure(state="disabled")

        ttk.Button(frame, text="Close", command=dialog.destroy).grid(row=2, column=0, sticky="e", pady=(10, 0))

    def _update_preview_button_states(self) -> None:
        if self.busy:
            self.stored_button.configure(state="disabled")
            return
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

    def open_roi_setup(self) -> None:
        if self.roi_dialog is not None and self.roi_dialog.dialog.winfo_exists():
            self.roi_dialog.dialog.focus_set()
            return

        initial_path: Path | None = None
        if self.current_preview_path:
            preview_path = Path(self.current_preview_path)
            if preview_path.exists():
                initial_path = preview_path

        def _on_roi_saved() -> None:
            self.status_var.set("Saved ROI updates")

        self.roi_dialog = ROISetupDialog(self.root, _on_roi_saved, initial_image_path=initial_path)

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

    def set_reference_from_config(self) -> None:
        if self.busy:
            return
        self.set_busy(True, "Capturing reference using saved config...")
        thread = threading.Thread(target=self._set_reference_thread, daemon=True)
        thread.start()

    def _set_reference_thread(self) -> None:
        active_paths = get_active_runtime_paths()
        config = read_json_file(active_paths["config_file"])
        result_code = set_reference(config)

        def finish() -> None:
            if result_code == 0:
                self.set_busy(False, "Reference saved")
                self.display_mode = "stored"
                self.refresh_view()
            else:
                self.set_busy(False, f"Set reference failed (exit {result_code})")
                self.refresh_view()

        self.root.after(0, finish)

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
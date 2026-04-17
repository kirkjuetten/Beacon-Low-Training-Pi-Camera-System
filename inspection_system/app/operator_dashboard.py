#!/usr/bin/env python3
"""Unified operator dashboard for Beacon inspection workflows."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageStat, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageStat = None
    ImageTk = None
    PIL_AVAILABLE = False

from inspection_system.app.camera_interface import (
    create_project,
    delete_project,
    export_project,
    get_active_runtime_paths,
    get_current_project,
    import_project,
    list_projects,
    switch_project,
)
from inspection_system.app.log_viewer import analyze_logs, load_training_logs
from inspection_system.app.runtime_controller import describe_edge_gate_status, describe_section_width_gate_status


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = REPO_ROOT / "inspection_system" / "app" / "capture_test.py"
PREVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
LIVE_PREVIEW_NAME = "dashboard_live_preview.png"
REFERENCE_PREVIEW_NAME = "golden_reference_image.png"
CONFIG_FIELD_SPECS = [
    ("capture.timeout_ms", "Capture Timeout (ms)", int),
    ("capture.shutter_us", "Shutter (us)", int),
    ("inspection.inspection_mode", "Inspection Mode", str),
    ("inspection.reference_strategy", "Reference Strategy", str),
    ("inspection.blend_mode", "Blend Mode", str),
    ("inspection.tolerance_mode", "Tolerance Mode", str),
    ("inspection.threshold_mode", "Threshold Mode", str),
    ("inspection.threshold_value", "Threshold Value", int),
    ("inspection.blur_kernel", "Blur Kernel (pixels)", int),
    ("inspection.reference_erode_iterations", "Reference Erode Iterations", int),
    ("inspection.reference_dilate_iterations", "Reference Dilate Iterations", int),
    ("inspection.sample_erode_iterations", "Sample Erode Iterations", int),
    ("inspection.sample_dilate_iterations", "Sample Dilate Iterations", int),
    ("inspection.min_feature_pixels", "Min Feature Pixels", int),
    ("inspection.min_required_coverage", "Min Required Coverage", float),
    ("inspection.max_outside_allowed_ratio", "Max Outside Allowed", float),
    ("inspection.min_section_coverage", "Min Section Coverage", float),
    ("inspection.max_mean_edge_distance_px", "Max Mean Edge Distance (px, optional)", float),
    ("inspection.max_section_edge_distance_px", "Max Section Edge Distance (px, optional)", float),
    ("inspection.max_section_width_delta_ratio", "Max Section Width Drift (ratio, optional)", float),
    ("inspection.min_ssim", "Min SSIM (optional)", float),
    ("inspection.max_mse", "Max MSE (optional)", float),
    ("inspection.min_anomaly_score", "Min Anomaly Score (optional)", float),
    ("inspection.image_display_mode", "Image Display Mode", str),
    ("inspection.save_debug_images", "Save Debug Images", bool),
    ("alignment.enabled", "Alignment Enabled", bool),
    ("alignment.tolerance_profile", "Alignment Profile", str),
    ("indicator_led.enabled", "Indicator LED Enabled", bool),
]

CONFIG_DROPDOWN_OPTIONS = {
    "inspection.inspection_mode": ["mask_only", "mask_and_ssim", "mask_and_ml", "full"],
    "inspection.reference_strategy": ["golden_only", "hybrid", "multi_good_experimental"],
    "inspection.blend_mode": ["hard_only", "blend_conservative", "blend_balanced", "blend_aggressive"],
    "inspection.tolerance_mode": ["strict", "balanced", "forgiving", "custom"],
    "inspection.threshold_mode": ["fixed", "fixed_inv", "otsu", "otsu_inv"],
    "inspection.image_display_mode": ["raw", "processed", "split"],
    "inspection.save_debug_images": ["True", "False"],
    "alignment.enabled": ["True", "False"],
    "alignment.tolerance_profile": ["strict", "balanced", "forgiving"],
    "indicator_led.enabled": ["True", "False"],
}

OPTIONAL_FLOAT_FIELDS = {
    "inspection.max_mean_edge_distance_px",
    "inspection.max_section_edge_distance_px",
    "inspection.max_section_width_delta_ratio",
    "inspection.min_ssim",
    "inspection.max_mse",
    "inspection.min_anomaly_score",
}

COMPACT_LAYOUT_MAX_WIDTH = 1100
COMPACT_LAYOUT_MAX_HEIGHT = 700


def read_json_file(file_path: Path) -> dict:
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json_file(file_path: Path, data: dict) -> None:
    file_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def get_nested_config_value(config: dict, dotted_path: str):
    current = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_nested_config_value(config: dict, dotted_path: str, value) -> None:
    parts = dotted_path.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def parse_config_value(raw_value: str, expected_type: type):
    text = raw_value.strip()
    if expected_type is bool:
        normalized = text.lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {raw_value}")
    if expected_type is int:
        return int(text)
    if expected_type is float:
        return float(text)
    return text


def apply_config_updates(config: dict, raw_updates: dict[str, str]) -> dict:
    updated = json.loads(json.dumps(config))
    for dotted_path, _, expected_type in CONFIG_FIELD_SPECS:
        if dotted_path not in raw_updates:
            continue
        raw_value = raw_updates[dotted_path]
        if dotted_path in OPTIONAL_FLOAT_FIELDS and raw_value.strip() == "":
            set_nested_config_value(updated, dotted_path, None)
            continue
        if raw_value.strip() == "":
            # Keep existing value unchanged when a field is blank in sparse legacy configs.
            continue
        set_nested_config_value(updated, dotted_path, parse_config_value(raw_value, expected_type))
    return updated


def build_config_editor_values(config: dict) -> dict[str, str]:
    values = {}
    for dotted_path, _, _ in CONFIG_FIELD_SPECS:
        value = get_nested_config_value(config, dotted_path)
        values[dotted_path] = "" if value is None else str(value)
    return values


def build_dashboard_hint_text(config: dict) -> str:
    edge_status_line, edge_hint = describe_edge_gate_status(config)
    width_status_line, width_hint = describe_section_width_gate_status(config)
    lines = [edge_status_line, width_status_line]
    if edge_hint:
        lines.append(edge_hint)
    if width_hint:
        lines.append(width_hint)
    if not edge_hint and not width_hint:
        lines.append("All geometry gates active.")
    return "\n".join(lines)


def is_informative_preview_image(image_path: Path) -> bool:
    """Return False for likely blank/black preview images."""
    if not PIL_AVAILABLE:
        return True
    try:
        image = Image.open(image_path).convert("L")
        min_px, max_px = image.getextrema()
        mean_px = float(ImageStat.Stat(image).mean[0])
    except Exception:
        return False

    contrast = int(max_px) - int(min_px)
    return contrast >= 6 and mean_px >= 6.0


def find_preview_image(reference_dir: Path, is_informative_fn=None) -> Path | None:
    if not reference_dir.exists():
        return None

    is_informative = is_informative_fn or is_informative_preview_image

    # Live preview is transient; no longer persisted
    # live_preview = reference_dir / LIVE_PREVIEW_NAME
    # if live_preview.exists() and is_informative(live_preview):
    #     return live_preview

    preferred = reference_dir / REFERENCE_PREVIEW_NAME
    if preferred.exists() and is_informative(preferred):
        return preferred

    candidates = [
        path for path in reference_dir.iterdir()
        if path.is_file() and path.suffix.lower() in PREVIEW_EXTENSIONS
    ]
    if not candidates:
        return None

    # Avoid showing debug diff/mask snapshots when a real sample image is available.
    non_debug = [
        path for path in candidates
        if not path.name.endswith("_diff.png") and not path.name.endswith("_mask.png")
    ]

    informative_non_debug = [path for path in non_debug if is_informative(path)]
    if informative_non_debug:
        return max(informative_non_debug, key=lambda path: path.stat().st_mtime)

    informative_candidates = [path for path in candidates if is_informative(path)]
    if informative_candidates:
        return max(informative_candidates, key=lambda path: path.stat().st_mtime)

    if non_debug:
        return max(non_debug, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def describe_preview_image(preview_path: Path) -> str:
    name = preview_path.name
    # Live preview is transient; no longer persisted
    # if name == LIVE_PREVIEW_NAME:
    #     return "live capture"
    if name == REFERENCE_PREVIEW_NAME:
        return "reference"
    if name.endswith("_diff.png"):
        return "difference debug"
    if name.endswith("_mask.png"):
        return "mask debug"
    return "latest sample"


def should_use_compact_layout(screen_width: int, screen_height: int) -> bool:
    """Return True when the screen is too small for the full dashboard layout."""
    return screen_width < COMPACT_LAYOUT_MAX_WIDTH or screen_height < COMPACT_LAYOUT_MAX_HEIGHT


def should_close_dashboard_on_launch(mode: str) -> bool:
    """Return whether dashboard should close after launching a tool."""
    return mode in {"project-manager"}


class OperatorDashboard:
    """Single-window operator UI for routine inspection tasks."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Beacon Inspection Dashboard")
        self.compact_layout = should_use_compact_layout(self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        self._configure_window_size()

        self.operation_running = False

        self.status_var = tk.StringVar(value="Ready")
        self.current_project_var = tk.StringVar(value="Current project: None")
        self.active_config_var = tk.StringVar(value="Config: -")
        self.active_reference_var = tk.StringVar(value="Reference: -")
        self.active_log_var = tk.StringVar(value="Logs: -")
        self.edge_gate_hint_var = tk.StringVar(value="Edge Gates: -")
        self.project_select_var = tk.StringVar()
        self.new_project_name_var = tk.StringVar()
        self.new_project_desc_var = tk.StringVar()

        self.summary_vars = {
            "total_samples": tk.StringVar(value="0"),
            "approved": tk.StringVar(value="0"),
            "rejected": tk.StringVar(value="0"),
            "reviewed": tk.StringVar(value="0"),
            "approval_rate": tk.StringVar(value="0.0%"),
            "sessions": tk.StringVar(value="0"),
        }

        self._build_layout()
        self.refresh_dashboard()

    def _configure_window_size(self) -> None:
        self.root.attributes("-fullscreen", True)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_padding = 10 if self.compact_layout else 14
        main = ttk.Frame(self.root, padding=main_padding)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        if not self.compact_layout:
            main.columnconfigure(1, weight=4, uniform="top_panels")
            main.columnconfigure(0, weight=5, uniform="top_panels")
            main.rowconfigure(1, weight=1)
            main.rowconfigure(2, weight=2)
        else:
            main.rowconfigure(2, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)

        header_font = ("Segoe UI", 15 if self.compact_layout else 18, "bold")
        ttk.Label(header, text="Beacon Operator Dashboard", font=header_font).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_var).grid(row=0, column=1, sticky="e")

        if self.compact_layout:
            self._build_compact_layout(main)
        else:
            self._build_standard_layout(main)

        status_bar = ttk.Label(self.root, textvariable=self.current_project_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky="ew")

    def _build_standard_layout(self, main: ttk.Frame) -> None:
        ops_frame = ttk.LabelFrame(main, text="Operations", padding=12)
        ops_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        ops_frame.columnconfigure(0, weight=1)
        ops_frame.columnconfigure(1, weight=1)

        project_frame = ttk.LabelFrame(main, text="Projects", padding=12)
        project_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 10), pady=(0, 10))
        project_frame.columnconfigure(0, weight=1)
        project_frame.columnconfigure(1, weight=1)

        output_frame = ttk.LabelFrame(main, text="Operator Console", padding=8)
        output_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        insights_frame = ttk.LabelFrame(main, text="Training Insights", padding=12)
        insights_frame.grid(row=2, column=1, sticky="nsew")
        insights_frame.columnconfigure(0, weight=1)
        insights_frame.rowconfigure(2, weight=1)

        self._build_operations_panel(ops_frame)
        self._build_project_panel(project_frame)
        self._build_console(output_frame)
        self._build_insights_panel(insights_frame)

    def _build_compact_layout(self, main: ttk.Frame) -> None:
        ops_frame = ttk.LabelFrame(main, text="Operations", padding=10)
        ops_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ops_frame.columnconfigure(0, weight=1)
        ops_frame.columnconfigure(1, weight=1)
        self._build_operations_panel(ops_frame)

        notebook = ttk.Notebook(main)
        notebook.grid(row=2, column=0, sticky="nsew")

        project_frame = ttk.Frame(notebook, padding=8)
        project_frame.columnconfigure(0, weight=1)
        project_frame.columnconfigure(1, weight=1)
        self._build_project_panel(project_frame)

        output_frame = ttk.Frame(notebook, padding=8)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        self._build_console(output_frame)

        insights_frame = ttk.Frame(notebook, padding=8)
        insights_frame.columnconfigure(0, weight=1)
        insights_frame.rowconfigure(2, weight=1)
        self._build_insights_panel(insights_frame)

        notebook.add(project_frame, text="Projects")
        notebook.add(output_frame, text="Console")
        notebook.add(insights_frame, text="Insights")

    def _build_operations_panel(self, parent: ttk.LabelFrame) -> None:
        info = ttk.Frame(parent)
        info.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8 if self.compact_layout else 12))
        info.columnconfigure(0, weight=1)

        ttk.Label(info, textvariable=self.active_config_var).grid(row=0, column=0, sticky="w", pady=1)
        ttk.Label(info, textvariable=self.active_reference_var).grid(row=1, column=0, sticky="w", pady=1)
        ttk.Label(info, textvariable=self.active_log_var).grid(row=2, column=0, sticky="w", pady=1)
        ttk.Label(info, textvariable=self.edge_gate_hint_var, wraplength=620 if self.compact_layout else 420, justify="left").grid(row=3, column=0, sticky="w", pady=(4, 1))

        self.capture_button = ttk.Button(parent, text="Capture Only", command=lambda: self.run_command("capture"))
        self.capture_button.grid(row=2, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.inspect_button = ttk.Button(parent, text="Inspect Part", command=lambda: self.run_command("inspect"))
        self.inspect_button.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.production_button = ttk.Button(parent, text="Launch Production", command=self.launch_production)
        self.production_button.grid(row=3, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.training_button = ttk.Button(parent, text="Launch Training", command=self.launch_training)
        self.training_button.grid(row=3, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.config_editor_button = ttk.Button(parent, text="Open Config + Preview", command=self.launch_config_editor)
        self.config_editor_button.grid(row=4, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.project_manager_button = ttk.Button(parent, text="Open Project Manager", command=self.launch_project_manager)
        self.project_manager_button.grid(row=4, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.refresh_button = ttk.Button(parent, text="Refresh Dashboard", command=self.refresh_dashboard)
        self.refresh_button.grid(row=5, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.exit_button = ttk.Button(parent, text="Exit Dashboard", command=self.exit_dashboard)
        self.exit_button.grid(row=5, column=1, sticky="ew", padx=(6, 0), pady=6)

    def _build_project_panel(self, parent: ttk.LabelFrame) -> None:
        parent.columnconfigure(0, weight=1)

        ttk.Label(parent, text="Active project").grid(row=0, column=0, sticky="w")
        self.project_combo = ttk.Combobox(parent, textvariable=self.project_select_var, state="readonly")
        self.project_combo.grid(row=1, column=0, sticky="ew", pady=(4, 10))

        ttk.Button(parent, text="Switch to Selected Project", command=self.switch_selected_project).grid(row=2, column=0, sticky="ew", pady=3)
        ttk.Separator(parent, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=12)
        ttk.Label(
            parent,
            text="Project creation, rename, delete, export, and import live in Project Manager.",
            wraplength=520 if self.compact_layout else 320,
            justify="left",
        ).grid(row=4, column=0, sticky="w")

    def _build_console(self, parent: ttk.LabelFrame) -> None:
        console_height = 12 if self.compact_layout else 18
        self.console = tk.Text(parent, wrap="word", height=console_height, bg="#111827", fg="#E5E7EB", insertbackground="#E5E7EB")
        self.console.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.console.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.console.configure(yscrollcommand=scrollbar.set)
        self.console.insert("end", "Beacon dashboard ready.\n")
        self.console.configure(state="disabled")

    def _build_insights_panel(self, parent: ttk.LabelFrame) -> None:
        metrics = ttk.Frame(parent)
        metrics.grid(row=0, column=0, sticky="ew")
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)

        items = [
            ("Total Samples", "total_samples"),
            ("Approved", "approved"),
            ("Rejected", "rejected"),
            ("Review", "reviewed"),
            ("Approval Rate", "approval_rate"),
            ("Sessions", "sessions"),
        ]

        for index, (label, key) in enumerate(items):
            row = index // 2
            column = (index % 2) * 2
            ttk.Label(metrics, text=label + ":", font=("Segoe UI", 10, "bold")).grid(row=row, column=column, sticky="w", pady=2)
            ttk.Label(metrics, textvariable=self.summary_vars[key]).grid(row=row, column=column + 1, sticky="w", pady=2)

        ttk.Label(parent, text="Recent log activity").grid(row=1, column=0, sticky="w", pady=(12, 6))
        self.recent_logs = tk.Listbox(parent, height=8 if self.compact_layout else 12)
        self.recent_logs.grid(row=2, column=0, sticky="nsew")

        buttons = ttk.Frame(parent)
        buttons.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="Refresh Logs", command=self.refresh_dashboard).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Clear Console", command=self.clear_console).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def append_console(self, text: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text)
        self.console.see("end")
        self.console.configure(state="disabled")

    def clear_console(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def set_busy(self, busy: bool, status: str) -> None:
        self.operation_running = busy
        self.status_var.set(status)
        state = "disabled" if busy else "normal"
        for button in [
            self.capture_button,
            self.inspect_button,
            self.production_button,
            self.training_button,
            self.project_manager_button,
            self.config_editor_button,
            self.refresh_button,
            self.exit_button,
        ]:
            button.configure(state=state)

    def exit_dashboard(self) -> None:
        if self.operation_running:
            messagebox.showinfo("Busy", "Wait for current operation to finish before exiting.")
            return
        self.root.destroy()

    def run_command(self, mode: str) -> None:
        if self.operation_running:
            messagebox.showinfo("Busy", "An operation is already running.")
            return

        self.append_console(f"\n> Running {mode}\n")
        self.set_busy(True, f"Running {mode}...")
        thread = threading.Thread(target=self._run_command_thread, args=(mode,), daemon=True)
        thread.start()

    def _run_command_thread(self, mode: str) -> None:
        process = subprocess.Popen(
            [sys.executable, str(CAPTURE_SCRIPT), mode],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        assert process.stdout is not None
        for line in process.stdout:
            self.root.after(0, self.append_console, line)

        return_code = process.wait()

        def finish() -> None:
            status = f"{mode} completed" if return_code == 0 else f"{mode} failed (exit {return_code})"
            self.set_busy(False, status)
            self.refresh_dashboard()

        self.root.after(0, finish)

    def launch_training(self) -> None:
        self._launch_tool("train", "training GUI")

    def launch_production(self) -> None:
        self._launch_tool("production", "production mode")

    def launch_project_manager(self) -> None:
        self._launch_tool("project-manager", "project manager")

    def launch_config_editor(self) -> None:
        self._launch_tool("config-editor", "config + preview")

    def _launch_tool(self, mode: str, label: str) -> None:
        try:
            process = subprocess.Popen(
                [sys.executable, str(CAPTURE_SCRIPT), mode],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.append_console(f"\n> Launched {label}.\n")
            self.status_var.set(f"Launched {label}")

            def monitor_output() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    self.root.after(0, self.append_console, f"[{mode}] {line}")
                return_code = process.wait()
                if return_code != 0:
                    self.root.after(0, self.append_console, f"> {label} exited with code {return_code}\n")
                    self.root.after(0, self.status_var.set, f"{label} failed (exit {return_code})")

            threading.Thread(target=monitor_output, daemon=True).start()

            if should_close_dashboard_on_launch(mode):
                if mode == "project-manager":
                    self.append_console("> Closing dashboard; use Project Manager -> Back to Dashboard when ready.\n")
                else:
                    self.append_console("> Closing dashboard; use Config + Preview -> Back to Dashboard when ready.\n")
                # Only close after we have confirmed child process is still running.
                def maybe_close() -> None:
                    if process.poll() is None:
                        self.root.destroy()
                    else:
                        self.append_console(f"> {label} failed to stay open; dashboard remains available.\n")

                self.root.after(500, maybe_close)
        except OSError as exc:
            messagebox.showerror("Launch failed", f"Could not launch {label}: {exc}")

    def switch_selected_project(self) -> None:
        project_name = self.project_select_var.get().strip()
        if not project_name:
            messagebox.showwarning("Project required", "Select a project first.")
            return
        if switch_project(project_name):
            self.append_console(f"\n> Switched to project {project_name}\n")
            self.refresh_dashboard()
        else:
            messagebox.showerror("Switch failed", f"Could not switch to project '{project_name}'.")

    def delete_selected_project(self) -> None:
        project_name = self.project_select_var.get().strip()
        if not project_name:
            messagebox.showwarning("Project required", "Select a project first.")
            return
        if not messagebox.askyesno("Delete Project", f"Delete project '{project_name}' and all of its files?"):
            return
        if delete_project(project_name):
            self.append_console(f"\n> Deleted project {project_name}\n")
            self.refresh_dashboard()
        else:
            messagebox.showerror("Delete failed", f"Could not delete project '{project_name}'.")

    def export_selected_project(self) -> None:
        project_name = self.project_select_var.get().strip()
        if not project_name:
            messagebox.showwarning("Project required", "Select a project first.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Export Project",
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
            initialfile=f"{project_name}_export",
        )
        if not file_path:
            return
        export_path = Path(file_path)
        if export_project(project_name, export_path):
            self.append_console(f"\n> Exported project {project_name} to {export_path}.zip\n")
        else:
            messagebox.showerror("Export failed", f"Could not export project '{project_name}'.")

    def import_project_from_zip(self) -> None:
        file_path = filedialog.askopenfilename(title="Import Project", filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")])
        if not file_path:
            return
        zip_path = Path(file_path)
        default_name = zip_path.stem
        existing_names = {project["name"] for project in list_projects()}

        project_name = default_name
        if project_name in existing_names:
            project_name = self.prompt_for_project_name(default_name + "_imported")
            if not project_name:
                return

        if import_project(zip_path, project_name):
            self.append_console(f"\n> Imported project {project_name} from {zip_path}\n")
            self.refresh_dashboard()
        else:
            messagebox.showerror("Import failed", f"Could not import project from '{zip_path}'.")

    def prompt_for_project_name(self, default_name: str) -> str | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Project Name")
        dialog.transient(self.root)
        dialog.grab_set()
        value = tk.StringVar(value=default_name)
        result = {"name": None}

        ttk.Label(dialog, text="Project name:").grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")
        entry = ttk.Entry(dialog, textvariable=value)
        entry.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        dialog.columnconfigure(0, weight=1)

        button_row = ttk.Frame(dialog)
        button_row.grid(row=2, column=0, padx=12, pady=(6, 12), sticky="ew")
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        def submit() -> None:
            result["name"] = value.get().strip() or None
            dialog.destroy()

        ttk.Button(button_row, text="Cancel", command=dialog.destroy).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(button_row, text="OK", command=submit).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        entry.focus_set()
        dialog.wait_window()
        return result["name"]

    def create_project_from_form(self) -> None:
        project_name = self.new_project_name_var.get().strip()
        description = self.new_project_desc_var.get().strip()
        if not project_name:
            messagebox.showwarning("Project required", "Enter a project name.")
            return
        if create_project(project_name, description):
            self.append_console(f"\n> Created project {project_name}\n")
            self.new_project_name_var.set("")
            self.new_project_desc_var.set("")
            self.refresh_dashboard()
        else:
            messagebox.showerror("Create failed", f"Could not create project '{project_name}'.")

    def refresh_dashboard(self) -> None:
        current_project = get_current_project() or "None"
        self.current_project_var.set(f"Current project: {current_project}")

        active_paths = get_active_runtime_paths()
        self.active_config_var.set(f"Config: {active_paths['config_file']}")
        self.active_reference_var.set(f"Reference: {active_paths['reference_dir']}")
        self.active_log_var.set(f"Logs: {active_paths['log_dir']}")
        active_config = read_json_file(active_paths["config_file"])
        self.edge_gate_hint_var.set(build_dashboard_hint_text(active_config))

        projects = list_projects()
        project_names = [project["name"] for project in projects]
        self.project_combo["values"] = project_names
        if current_project != "None" and current_project in project_names:
            self.project_select_var.set(current_project)
        elif project_names:
            self.project_select_var.set(project_names[0])
        else:
            self.project_select_var.set("")

        logs = load_training_logs(active_paths["log_dir"])
        summary = analyze_logs(logs) if logs else {}
        self.summary_vars["total_samples"].set(str(summary.get("total_samples", 0)))
        self.summary_vars["approved"].set(str(summary.get("approved", 0)))
        self.summary_vars["rejected"].set(str(summary.get("rejected", 0)))
        self.summary_vars["reviewed"].set(str(summary.get("reviewed", 0)))
        self.summary_vars["approval_rate"].set(f"{summary.get('approval_rate', 0):.1%}")
        self.summary_vars["sessions"].set(str(summary.get("sessions", 0)))

        self.recent_logs.delete(0, tk.END)
        for log in logs[-12:]:
            self.recent_logs.insert(tk.END, f"[{log['timestamp']}] {log['status']} -> {log['feedback']} | {log['filename']}")

def main() -> None:
    root = tk.Tk()
    OperatorDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
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
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    Image = None
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


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = REPO_ROOT / "inspection_system" / "app" / "capture_test.py"
PREVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
CONFIG_FIELD_SPECS = [
    ("capture.timeout_ms", "Capture Timeout (ms)", int),
    ("capture.shutter_us", "Shutter (us)", int),
    ("inspection.threshold_value", "Threshold Value", int),
    ("inspection.min_feature_pixels", "Min Feature Pixels", int),
    ("inspection.min_required_coverage", "Min Required Coverage", float),
    ("inspection.max_outside_allowed_ratio", "Max Outside Allowed", float),
    ("inspection.min_section_coverage", "Min Section Coverage", float),
    ("inspection.min_ssim", "Min SSIM (optional)", float),
    ("inspection.max_mse", "Max MSE (optional)", float),
    ("inspection.min_anomaly_score", "Min Anomaly Score (optional)", float),
    ("inspection.save_debug_images", "Save Debug Images", bool),
    ("alignment.enabled", "Alignment Enabled", bool),
    ("indicator_led.enabled", "Indicator LED Enabled", bool),
]

OPTIONAL_FLOAT_FIELDS = {
    "inspection.min_ssim",
    "inspection.max_mse",
    "inspection.min_anomaly_score",
}


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
        set_nested_config_value(updated, dotted_path, parse_config_value(raw_value, expected_type))
    return updated


def build_config_editor_values(config: dict) -> dict[str, str]:
    values = {}
    for dotted_path, _, _ in CONFIG_FIELD_SPECS:
        value = get_nested_config_value(config, dotted_path)
        values[dotted_path] = "" if value is None else str(value)
    return values


def find_preview_image(reference_dir: Path) -> Path | None:
    if not reference_dir.exists():
        return None

    candidates = [
        path for path in reference_dir.iterdir()
        if path.is_file() and path.suffix.lower() in PREVIEW_EXTENSIONS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


class OperatorDashboard:
    """Single-window operator UI for routine inspection tasks."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Beacon Inspection Dashboard")
        self.root.geometry("1360x860")
        self.root.minsize(1180, 760)

        self.operation_running = False
        self.preview_photo = None
        self.config_vars: dict[str, tk.StringVar] = {}

        self.status_var = tk.StringVar(value="Ready")
        self.current_project_var = tk.StringVar(value="Current project: None")
        self.active_config_var = tk.StringVar(value="Config: -")
        self.active_reference_var = tk.StringVar(value="Reference: -")
        self.active_log_var = tk.StringVar(value="Logs: -")
        self.project_select_var = tk.StringVar()
        self.new_project_name_var = tk.StringVar()
        self.new_project_desc_var = tk.StringVar()
        self.preview_path_var = tk.StringVar(value="Preview: none")

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

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=3)
        main.columnconfigure(2, weight=2)
        main.rowconfigure(1, weight=1)
        main.rowconfigure(2, weight=2)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Beacon Operator Dashboard", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_var).grid(row=0, column=1, sticky="e")

        ops_frame = ttk.LabelFrame(main, text="Operations", padding=12)
        ops_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        ops_frame.columnconfigure(0, weight=1)
        ops_frame.columnconfigure(1, weight=1)

        project_frame = ttk.LabelFrame(main, text="Projects", padding=12)
        project_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 10), pady=(0, 10))
        project_frame.columnconfigure(0, weight=1)
        project_frame.columnconfigure(1, weight=1)

        preview_frame = ttk.LabelFrame(main, text="Latest Preview", padding=12)
        preview_frame.grid(row=1, column=2, sticky="nsew", pady=(0, 10))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        output_frame = ttk.LabelFrame(main, text="Operator Console", padding=8)
        output_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=(0, 10))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        insights_frame = ttk.LabelFrame(main, text="Training Insights", padding=12)
        insights_frame.grid(row=2, column=2, sticky="nsew")
        insights_frame.columnconfigure(0, weight=1)
        insights_frame.rowconfigure(2, weight=1)

        self._build_operations_panel(ops_frame)
        self._build_project_panel(project_frame)
        self._build_preview_panel(preview_frame)
        self._build_console(output_frame)
        self._build_insights_panel(insights_frame)

        status_bar = ttk.Label(self.root, textvariable=self.current_project_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky="ew")

    def _build_operations_panel(self, parent: ttk.LabelFrame) -> None:
        info = ttk.Frame(parent)
        info.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        info.columnconfigure(0, weight=1)

        ttk.Label(info, textvariable=self.active_config_var).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(info, textvariable=self.active_reference_var).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(info, textvariable=self.active_log_var).grid(row=2, column=0, sticky="w", pady=2)

        self.capture_button = ttk.Button(parent, text="Capture Only", command=lambda: self.run_command("capture"))
        self.capture_button.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.reference_button = ttk.Button(parent, text="Set Reference", command=lambda: self.run_command("set-reference"))
        self.reference_button.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.inspect_button = ttk.Button(parent, text="Inspect Part", command=lambda: self.run_command("inspect"))
        self.inspect_button.grid(row=2, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.training_button = ttk.Button(parent, text="Launch Training", command=self.launch_training)
        self.training_button.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.project_manager_button = ttk.Button(parent, text="Open Project Manager", command=self.launch_project_manager)
        self.project_manager_button.grid(row=3, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.refresh_button = ttk.Button(parent, text="Refresh Dashboard", command=self.refresh_dashboard)
        self.refresh_button.grid(row=3, column=1, sticky="ew", padx=(6, 0), pady=6)

        config_frame = ttk.LabelFrame(parent, text="Config Editor", padding=8)
        config_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        config_frame.columnconfigure(1, weight=1)

        for row, (dotted_path, label, _) in enumerate(CONFIG_FIELD_SPECS):
            ttk.Label(config_frame, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))
            var = tk.StringVar()
            self.config_vars[dotted_path] = var
            ttk.Entry(config_frame, textvariable=var).grid(row=row, column=1, sticky="ew", pady=3)

        buttons = ttk.Frame(config_frame)
        buttons.grid(row=len(CONFIG_FIELD_SPECS), column=0, columnspan=2, sticky="ew", pady=(10, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="Reload Config", command=self.reload_config_editor).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Save Config", command=self.save_config_editor).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _build_project_panel(self, parent: ttk.LabelFrame) -> None:
        ttk.Label(parent, text="Active project").grid(row=0, column=0, columnspan=2, sticky="w")
        self.project_combo = ttk.Combobox(parent, textvariable=self.project_select_var, state="readonly")
        self.project_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 8))

        ttk.Button(parent, text="Switch", command=self.switch_selected_project).grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=4)
        ttk.Button(parent, text="Delete", command=self.delete_selected_project).grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=4)
        ttk.Button(parent, text="Export", command=self.export_selected_project).grid(row=3, column=0, sticky="ew", padx=(0, 4), pady=4)
        ttk.Button(parent, text="Import", command=self.import_project_from_zip).grid(row=3, column=1, sticky="ew", padx=(4, 0), pady=4)

        ttk.Separator(parent, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=12)

        ttk.Label(parent, text="Create project").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Entry(parent, textvariable=self.new_project_name_var).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        ttk.Entry(parent, textvariable=self.new_project_desc_var).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Button(parent, text="Create Project", command=self.create_project_from_form).grid(row=8, column=0, columnspan=2, sticky="ew")

    def _build_preview_panel(self, parent: ttk.LabelFrame) -> None:
        ttk.Label(parent, textvariable=self.preview_path_var, wraplength=280).grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(parent, text="No preview image available", anchor="center")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        ttk.Button(parent, text="Refresh Preview", command=self.refresh_dashboard).grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def _build_console(self, parent: ttk.LabelFrame) -> None:
        self.console = tk.Text(parent, wrap="word", height=18, bg="#111827", fg="#E5E7EB", insertbackground="#E5E7EB")
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
        self.recent_logs = tk.Listbox(parent, height=12)
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
        for button in [self.capture_button, self.reference_button, self.inspect_button, self.training_button, self.project_manager_button, self.refresh_button]:
            button.configure(state=state)

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

    def launch_project_manager(self) -> None:
        self._launch_tool("project-manager", "project manager")

    def _launch_tool(self, mode: str, label: str) -> None:
        try:
            subprocess.Popen([sys.executable, str(CAPTURE_SCRIPT), mode], cwd=str(REPO_ROOT))
            self.append_console(f"\n> Launched {label}.\n")
            self.status_var.set(f"Launched {label}")
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
            self.append_console(f"\n> Saved config updates to {config_path}\n")
            self.refresh_dashboard()
        except ValueError as exc:
            messagebox.showerror("Invalid config value", str(exc))

    def refresh_preview(self, active_paths: dict[str, Path]) -> None:
        preview_path = find_preview_image(active_paths["reference_dir"])
        if preview_path is None:
            self.preview_path_var.set("Preview: none")
            self.preview_label.configure(text="No preview image available", image="")
            self.preview_photo = None
            return

        self.preview_path_var.set(f"Preview: {preview_path.name}")
        if PIL_AVAILABLE:
            image = Image.open(preview_path)
            image.thumbnail((320, 240))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_photo, text="")
        else:
            self.preview_label.configure(text=str(preview_path), image="")
            self.preview_photo = None

    def refresh_dashboard(self) -> None:
        current_project = get_current_project() or "None"
        self.current_project_var.set(f"Current project: {current_project}")

        active_paths = get_active_runtime_paths()
        self.active_config_var.set(f"Config: {active_paths['config_file']}")
        self.active_reference_var.set(f"Reference: {active_paths['reference_dir']}")
        self.active_log_var.set(f"Logs: {active_paths['log_dir']}")

        projects = list_projects()
        project_names = [project["name"] for project in projects]
        self.project_combo["values"] = project_names
        if current_project != "None" and current_project in project_names:
            self.project_select_var.set(current_project)
        elif project_names:
            self.project_select_var.set(project_names[0])
        else:
            self.project_select_var.set("")

        self.reload_config_editor()
        self.refresh_preview(active_paths)

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
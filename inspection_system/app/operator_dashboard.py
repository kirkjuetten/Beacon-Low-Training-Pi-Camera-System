#!/usr/bin/env python3
"""Unified operator dashboard for Beacon inspection workflows."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from inspection_system.app.camera_interface import (
    create_project,
    delete_project,
    export_project,
    get_active_runtime_paths,
    get_current_project,
    import_project,
    list_projects,
    load_config,
    switch_project,
)
from inspection_system.app.command_runner import launch_monitored_command, stream_command
from inspection_system.app.config_service import read_json_file
from inspection_system.app.dataset_capture import TestDataCollectorDialog
from inspection_system.app.log_viewer import analyze_logs, load_training_logs
from inspection_system.app.preview_service import describe_preview_image, find_preview_image
from inspection_system.app.runtime_controller import describe_edge_gate_status, describe_section_center_gate_status, describe_section_width_gate_status
from inspection_system.app.runtime_controller import format_commissioning_status_lines, get_commissioning_status
from inspection_system.app.scrollable_frame import VerticalScrolledFrame


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = REPO_ROOT / "inspection_system" / "app" / "capture_test.py"

COMPACT_LAYOUT_MAX_WIDTH = 1100
COMPACT_LAYOUT_MAX_HEIGHT = 700


def build_dashboard_hint_text(config: dict, active_paths: Path | dict | None = None) -> str:
    edge_status_line, edge_hint = describe_edge_gate_status(config)
    width_status_line, width_hint = describe_section_width_gate_status(config)
    center_status_line, center_hint = describe_section_center_gate_status(config)
    commissioning_lines = []
    if isinstance(active_paths, dict):
        commissioning_lines = format_commissioning_status_lines(get_commissioning_status(config, active_paths))
    lines = commissioning_lines + [edge_status_line, width_status_line, center_status_line]
    if edge_hint:
        lines.append(edge_hint)
    if width_hint:
        lines.append(width_hint)
    if center_hint:
        lines.append(center_hint)
    if not edge_hint and not width_hint and not center_hint:
        lines.append("All geometry gates active.")
    return "\n".join(lines)


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
        self.test_data_dialog = None

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
        shell = VerticalScrolledFrame(self.root, content_padding=main_padding)
        shell.grid(row=0, column=0, sticky="nsew")
        main = shell.content
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
        self.exit_button = ttk.Button(parent, text="Exit Dashboard", command=self.exit_dashboard)
        self.exit_button.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.production_button = ttk.Button(parent, text="Launch Production", command=self.launch_production)
        self.production_button.grid(row=3, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.training_button = ttk.Button(parent, text="Launch Training", command=self.launch_training)
        self.training_button.grid(row=3, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.config_editor_button = ttk.Button(parent, text="Open Config + Preview", command=self.launch_config_editor)
        self.config_editor_button.grid(row=4, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.test_data_button = ttk.Button(parent, text="Collect Test Images", command=self.launch_test_data_collector)
        self.test_data_button.grid(row=4, column=1, sticky="ew", padx=(6, 0), pady=6)
        self.project_manager_button = ttk.Button(parent, text="Open Project Manager", command=self.launch_project_manager)
        self.project_manager_button.grid(row=5, column=0, sticky="ew", padx=(0, 6), pady=6)
        self.refresh_button = ttk.Button(parent, text="Refresh Dashboard", command=self.refresh_dashboard)
        self.refresh_button.grid(row=5, column=1, sticky="ew", padx=(6, 0), pady=6)

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
            self.production_button,
            self.training_button,
            self.test_data_button,
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
        command = [sys.executable, str(CAPTURE_SCRIPT), mode]

        def finish(return_code: int) -> None:
            status = f"{mode} completed" if return_code == 0 else f"{mode} failed (exit {return_code})"
            self.set_busy(False, status)
            self.refresh_dashboard()

        def handle_error(exc: OSError) -> None:
            self.set_busy(False, f"{mode} failed to start")
            messagebox.showerror("Run failed", f"Could not run {mode}: {exc}")

        stream_command(
            command,
            cwd=REPO_ROOT,
            on_output=lambda line: self.root.after(0, self.append_console, line),
            on_complete=lambda return_code: self.root.after(0, finish, return_code),
            on_error=lambda exc: self.root.after(0, handle_error, exc),
        )

    def launch_training(self) -> None:
        self._launch_tool("train", "training GUI")

    def launch_production(self) -> None:
        self._launch_tool("production", "production mode")

    def launch_project_manager(self) -> None:
        self._launch_tool("project-manager", "project manager")

    def launch_config_editor(self) -> None:
        self._launch_tool("config-editor", "config + preview")

    def launch_test_data_collector(self) -> None:
        if self.operation_running:
            messagebox.showinfo("Busy", "Wait for current operation to finish before opening test image collection.")
            return
        if self.test_data_dialog is not None and self.test_data_dialog.is_open():
            self.test_data_dialog.focus()
            return
        self.test_data_dialog = TestDataCollectorDialog(
            self.root,
            config_loader=load_config,
            active_paths_loader=get_active_runtime_paths,
        )

    def _launch_tool(self, mode: str, label: str) -> None:
        try:
            process = launch_monitored_command(
                [sys.executable, str(CAPTURE_SCRIPT), mode],
                cwd=REPO_ROOT,
                on_output=lambda line: self.root.after(0, self.append_console, f"[{mode}] {line}"),
                on_exit=lambda return_code: self.root.after(0, self._handle_launched_tool_exit, mode, label, return_code),
            )
            self.append_console(f"\n> Launched {label}.\n")
            self.status_var.set(f"Launched {label}")

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

    def _handle_launched_tool_exit(self, mode: str, label: str, return_code: int) -> None:
        if return_code != 0:
            self.append_console(f"> {label} exited with code {return_code}\n")
            self.status_var.set(f"{label} failed (exit {return_code})")

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
        self.edge_gate_hint_var.set(build_dashboard_hint_text(active_config, active_paths))

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
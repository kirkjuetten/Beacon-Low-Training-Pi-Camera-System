#!/usr/bin/env python3
"""
Project Manager GUI for Beacon Inspection System

Provides a graphical interface for managing multiple inspection projects.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from inspection_system.app.camera_interface import (
    create_project, switch_project, get_current_project, list_projects,
    delete_project, export_project, import_project, PROJECTS_DIR
)
from inspection_system.app.touch_keyboard import TouchKeyboardManager


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SCRIPT = REPO_ROOT / "inspection_system" / "app" / "capture_test.py"


class ProjectManagerGUI:
    """GUI for managing inspection projects."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Beacon Inspection - Project Manager")
        self._configure_window_size()
        self.keyboard_manager = TouchKeyboardManager(self.root)

        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Project Manager",
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Current project display
        current_frame = ttk.LabelFrame(main_frame, text="Current Project", padding="5")
        current_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.current_label = ttk.Label(current_frame, text="Loading...")
        self.current_label.grid(row=0, column=0, sticky=tk.W)

        # Projects list
        list_frame = ttk.LabelFrame(main_frame, text="Projects", padding="5")
        list_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Treeview for projects
        columns = ("Name", "Description", "Created", "Status")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        # Left buttons
        left_buttons = ttk.Frame(buttons_frame)
        left_buttons.grid(row=0, column=0, padx=(0, 10))

        ttk.Button(left_buttons, text="Create Project",
                  command=self.create_project_dialog).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(left_buttons, text="Switch To",
                  command=self.switch_to_project).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(left_buttons, text="Delete",
                  command=self.delete_project).grid(row=0, column=2)

        # Right buttons
        right_buttons = ttk.Frame(buttons_frame)
        right_buttons.grid(row=0, column=1)

        ttk.Button(right_buttons, text="Export",
                  command=self.export_project).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(right_buttons, text="Import",
                  command=self.import_project).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(right_buttons, text="Refresh",
                  command=self.refresh_projects).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(right_buttons, text="Back to Dashboard",
                  command=self._launch_dashboard).grid(row=0, column=3, padx=(10, 0))
        if self.keyboard_manager.enabled:
            ttk.Button(right_buttons, text="Hide Keyboard",
                      command=self.keyboard_manager.hide_keyboard).grid(row=0, column=4, padx=(10, 0))
        ttk.Button(right_buttons, text="Exit",
                  command=self.root.destroy).grid(row=0, column=5, padx=(10, 0))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # Bind events
        self.tree.bind('<Double-1>', self.on_project_double_click)

        # Load initial data
        self.refresh_projects()

    def _configure_window_size(self) -> None:
        self.root.update_idletasks()
        screen_w = int(self.root.winfo_screenwidth())
        screen_h = int(self.root.winfo_screenheight())

        target_w = min(980, max(720, screen_w - 20))
        target_h = min(680, max(420, screen_h - 90))
        self.root.geometry(f"{target_w}x{target_h}+8+8")

        min_w = min(720, max(620, screen_w - 30))
        min_h = min(420, max(360, screen_h - 120))
        self.root.minsize(min_w, min_h)

    def _launch_dashboard(self):
        """Close project manager and open the operator dashboard."""
        self.root.destroy()
        from inspection_system.app.operator_dashboard import main as dashboard_main
        dashboard_main()

    def refresh_projects(self):
        """Refresh the projects list."""
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Get current project
            current = get_current_project()
            self.current_label.config(text=f"Current: {current or 'None'}")

            # Load projects
            projects = list_projects()
            for project in projects:
                status = "Active" if project["is_current"] else ""
                self.tree.insert("", tk.END, values=(
                    project["name"],
                    project["description"],
                    project["created"],
                    status
                ))

            self.status_var.set(f"Loaded {len(projects)} projects")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load projects: {e}")
            self.status_var.set("Error loading projects")

    def create_project_dialog(self):
        """Show dialog to create a new project."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create New Project")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.geometry("+{}+{}".format(
            self.root.winfo_rootx() + 50,
            self.root.winfo_rooty() + 50
        ))

        # Content
        ttk.Label(dialog, text="Project Name:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        name_entry.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(dialog, text="Description:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        desc_var = tk.StringVar()
        desc_entry = ttk.Entry(dialog, textvariable=desc_var, width=30)
        desc_entry.grid(row=1, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)

        def create():
            name = name_var.get().strip()
            desc = desc_var.get().strip()

            if not name:
                messagebox.showerror("Error", "Project name is required")
                return

            if create_project(name, desc):
                self.refresh_projects()
                dialog.destroy()
                messagebox.showinfo("Success", f"Project '{name}' created successfully")
            else:
                messagebox.showerror("Error", f"Failed to create project '{name}'")

        ttk.Button(button_frame, text="Create", command=create).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=5)

        name_entry.focus()
        dialog.bind('<Return>', lambda e: create())
        dialog.bind('<Escape>', lambda e: dialog.destroy())

    def switch_to_project(self):
        """Switch to the selected project."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a project first")
            return

        item = self.tree.item(selection[0])
        project_name = str(item["values"][0]).strip()

        if switch_project(project_name):
            self.refresh_projects()
            messagebox.showinfo("Success", f"Switched to project '{project_name}'")
        else:
            messagebox.showerror("Error", f"Failed to switch to project '{project_name}'")

    def delete_project(self):
        """Delete the selected project."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a project first")
            return

        item = self.tree.item(selection[0])
        project_name = item['values'][0]

        if messagebox.askyesno("Confirm Delete",
                              f"Are you sure you want to delete project '{project_name}'?\n\n"
                              "This will permanently delete all project files and cannot be undone."):
            if delete_project(project_name):
                self.refresh_projects()
                messagebox.showinfo("Success", f"Project '{project_name}' deleted")
            else:
                messagebox.showerror("Error", f"Failed to delete project '{project_name}'")

    def export_project(self):
        """Export the selected project."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a project first")
            return

        item = self.tree.item(selection[0])
        project_name = item['values'][0]

        # Ask for export location
        file_path = filedialog.asksaveasfilename(
            title="Export Project",
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
            initialfile=f"{project_name}_export"
        )

        if file_path:
            export_path = Path(file_path)
            if export_project(project_name, export_path):
                messagebox.showinfo("Success", f"Project exported to {export_path}.zip")
            else:
                messagebox.showerror("Error", "Failed to export project")

    def import_project(self):
        """Import a project from ZIP file."""
        file_path = filedialog.askopenfilename(
            title="Import Project",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
        )

        if file_path:
            zip_path = Path(file_path)
            project_name = zip_path.stem

            # Check if project already exists
            projects = list_projects()
            existing_names = [p["name"] for p in projects]

            if project_name in existing_names:
                # Ask for new name
                dialog = tk.Toplevel(self.root)
                dialog.title("Project Name Conflict")
                dialog.geometry("400x150")

                ttk.Label(dialog, text=f"Project '{project_name}' already exists.\nEnter a new name:").grid(row=0, column=0, columnspan=2, padx=10, pady=10)

                name_var = tk.StringVar(value=f"{project_name}_imported")
                name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
                name_entry.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

                def do_import():
                    new_name = name_var.get().strip()
                    if not new_name:
                        messagebox.showerror("Error", "Project name is required")
                        return

                    if import_project(zip_path, new_name):
                        self.refresh_projects()
                        dialog.destroy()
                        messagebox.showinfo("Success", f"Project imported as '{new_name}'")
                    else:
                        messagebox.showerror("Error", "Failed to import project")

                button_frame = ttk.Frame(dialog)
                button_frame.grid(row=2, column=0, columnspan=2, pady=10)
                ttk.Button(button_frame, text="Import", command=do_import).grid(row=0, column=0, padx=5)
                ttk.Button(button_frame, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=5)

                name_entry.focus()
                dialog.transient(self.root)
                dialog.grab_set()
            else:
                if import_project(zip_path):
                    self.refresh_projects()
                    messagebox.showinfo("Success", f"Project '{project_name}' imported")
                else:
                    messagebox.showerror("Error", "Failed to import project")

    def on_project_double_click(self, event):
        """Handle double-click on project (switch to it)."""
        self.switch_to_project()


def launch_training_gui():
    """Launch the interactive training GUI."""
    if not PYGAME_AVAILABLE:
        messagebox.showerror("Error", "Pygame is required for training GUI.\nInstall with: pip install pygame")
        return
    try:
        subprocess.Popen([sys.executable, str(CAPTURE_SCRIPT), "train"], cwd=str(REPO_ROOT))
    except OSError as exc:
        messagebox.showerror("Error", f"Failed to launch training GUI: {exc}")


def main():
    """Main entry point for the project manager."""
    root = tk.Tk()
    app = ProjectManagerGUI(root)

    # Add menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Refresh", command=app.refresh_projects)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.destroy)

    # Tools menu
    tools_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Tools", menu=tools_menu)
    tools_menu.add_command(label="Launch Training GUI", command=launch_training_gui)
    tools_menu.add_separator()
    tools_menu.add_command(label="Launch Dashboard", command=app._launch_dashboard)

    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
        "About", "Beacon Inspection System - Project Manager\n\n"
        "Manage multiple inspection projects with separate configurations\n"
        "and reference images for different products or setups."))

    root.mainloop()


if __name__ == "__main__":
    main()

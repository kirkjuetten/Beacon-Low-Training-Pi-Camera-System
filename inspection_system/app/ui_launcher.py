#!/usr/bin/env python3


def launch_project_manager() -> int:
    try:
        from inspection_system.app.project_manager import main as pm_main

        pm_main()
        return 0
    except ImportError as exc:
        print(f"GUI not available: {exc}")
        print("Install tkinter and pygame for GUI support")
        return 1


def launch_dashboard() -> int:
    try:
        from inspection_system.app.operator_dashboard import main as dashboard_main

        dashboard_main()
        return 0
    except ImportError as exc:
        print(f"Dashboard not available: {exc}")
        print("Install tkinter for dashboard support")
        return 1

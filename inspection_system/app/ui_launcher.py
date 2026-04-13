#!/usr/bin/env python3


def _is_tk_display_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "TclError"


def launch_project_manager() -> int:
    try:
        from inspection_system.app.project_manager import main as pm_main

        pm_main()
        return 0
    except ImportError as exc:
        print(f"GUI not available: {exc}")
        print("Install tkinter and pygame for GUI support")
        return 1
    except Exception as exc:
        if _is_tk_display_error(exc):
            print("Project manager requires a graphical desktop session.")
            print("Run this on the Pi desktop (monitor attached) or via VNC.")
            return 2
        print(f"Project manager failed to launch: {exc}")
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
    except Exception as exc:
        if _is_tk_display_error(exc):
            print("Dashboard requires a graphical desktop session.")
            print("Run this on the Pi desktop (monitor attached) or via VNC.")
            print("For SSH-only operation, use:")
            print("  python3 -m inspection_system.app.capture_test capture")
            print("  python3 -m inspection_system.app.capture_test inspect")
            return 2
        print(f"Dashboard failed to launch: {exc}")
        return 1

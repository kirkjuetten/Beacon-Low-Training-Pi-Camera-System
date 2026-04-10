#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inspection_system.app.camera_interface import IndicatorLED, load_config
from inspection_system.app.project_service import (
    handle_create_project,
    handle_list_projects,
    handle_switch_project,
)
from inspection_system.app.reference_service import save_debug_outputs, set_reference
from inspection_system.app.runtime_controller import (
    run_capture_and_inspect,
    run_capture_only,
    run_interactive_training,
)
from inspection_system.app.ui_launcher import launch_dashboard, launch_project_manager


def main() -> int:
    config = load_config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"

    led_cfg = config.get("indicator_led", {})
    indicator = IndicatorLED(
        enabled=bool(led_cfg.get("enabled", False)),
        pass_gpio=int(led_cfg.get("pass_gpio", 23)),
        fail_gpio=int(led_cfg.get("fail_gpio", 24)),
        pulse_ms=int(led_cfg.get("pulse_ms", 750)),
    )

    try:
        if mode == "capture":
            return run_capture_only(config)
        if mode == "set-reference":
            return set_reference(config)
        if mode == "inspect":
            return run_capture_and_inspect(config, indicator)
        if mode == "train":
            return run_interactive_training(config)

        if mode == "create-project":
            return handle_create_project(sys.argv)

        if mode == "switch-project":
            return handle_switch_project(sys.argv)

        if mode == "list-projects":
            return handle_list_projects()

        if mode == "project-manager":
            return launch_project_manager()

        if mode == "dashboard":
            return launch_dashboard()

        print("Usage:")
        print("  python3 capture_test.py capture")
        print("  python3 capture_test.py set-reference")
        print("  python3 capture_test.py inspect")
        print("  python3 capture_test.py train")
        print("  python3 capture_test.py dashboard")
        print("  python3 capture_test.py create-project <name> [description]")
        print("  python3 capture_test.py switch-project <name>")
        print("  python3 capture_test.py list-projects")
        print("  python3 capture_test.py project-manager  # GUI")
        return 2
    finally:
        indicator.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

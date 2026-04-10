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


def print_usage() -> None:
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


def command_capture(config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return run_capture_only(config)


def command_set_reference(config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return set_reference(config)


def command_inspect(config: dict, indicator: IndicatorLED, _argv: list[str]) -> int:
    return run_capture_and_inspect(config, indicator)


def command_train(config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return run_interactive_training(config)


def command_create_project(_config: dict, _indicator: IndicatorLED, argv: list[str]) -> int:
    return handle_create_project(argv)


def command_switch_project(_config: dict, _indicator: IndicatorLED, argv: list[str]) -> int:
    return handle_switch_project(argv)


def command_list_projects(_config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return handle_list_projects()


def command_project_manager(_config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return launch_project_manager()


def command_dashboard(_config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return launch_dashboard()


COMMAND_HANDLERS = {
    "capture": command_capture,
    "set-reference": command_set_reference,
    "inspect": command_inspect,
    "train": command_train,
    "create-project": command_create_project,
    "switch-project": command_switch_project,
    "list-projects": command_list_projects,
    "project-manager": command_project_manager,
    "dashboard": command_dashboard,
}


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
        handler = COMMAND_HANDLERS.get(mode)
        if handler is None:
            print_usage()
            return 2
        return handler(config, indicator, sys.argv)
    finally:
        indicator.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

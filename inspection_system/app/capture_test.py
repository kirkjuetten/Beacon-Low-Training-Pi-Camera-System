#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inspection_system.app.camera_interface import IndicatorLED, load_config
from inspection_system.app.indicator_context import IndicatorContext
from inspection_system.app.project_service import (
    handle_create_project,
    handle_list_projects,
    handle_switch_project,
)
from inspection_system.app.pilot_readiness import print_supervised_pilot_report
from inspection_system.app.reference_service import save_debug_outputs, set_reference
from inspection_system.app.runtime_controller import (
    run_capture_and_inspect,
    run_capture_only,
    run_interactive_training,
    run_production_mode,
)
from inspection_system.app.ui_launcher import launch_config_editor, launch_dashboard, launch_project_manager


def print_usage() -> None:
    print("Usage:")
    print("  python3 capture_test.py dashboard        # Operator home (default)")
    print("  python3 capture_test.py pilot-readiness  # Check technical readiness for supervised pilot")
    print("  python3 capture_test.py quick-check      # Run functional smoke sequence")
    print("  python3 capture_test.py capture          # Manual capture only")
    print("  python3 capture_test.py set-reference    # Set reference image")
    print("  python3 capture_test.py inspect          # Run inspection on new part")
    print("  python3 capture_test.py train            # Launch training workflow")
    print("  python3 capture_test.py production       # Launch production inspection screen")
    print("  python3 capture_test.py create-project <name> [description]")
    print("  python3 capture_test.py switch-project <name>")
    print("  python3 capture_test.py list-projects")
    print("  python3 capture_test.py project-manager  # GUI project manager")
    print("  python3 capture_test.py config-editor    # GUI config + preview")


def command_capture(config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return run_capture_only(config)


def command_set_reference(config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return set_reference(config)


def command_inspect(config: dict, indicator: IndicatorLED, _argv: list[str]) -> int:
    return run_capture_and_inspect(config, indicator)


def command_train(config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return run_interactive_training(config)


def command_production(config: dict, indicator: IndicatorLED, _argv: list[str]) -> int:
    return run_production_mode(config, indicator)


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


def command_pilot_readiness(config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return print_supervised_pilot_report(config)


def command_config_editor(_config: dict, _indicator: IndicatorLED, _argv: list[str]) -> int:
    return launch_config_editor()


def command_quick_check(config: dict, indicator: IndicatorLED, _argv: list[str]) -> int:
    """Run core functional checks in sequence and summarize results."""
    steps = [
        ("list-projects", lambda: handle_list_projects()),
        ("capture", lambda: run_capture_only(config)),
        ("inspect", lambda: run_capture_and_inspect(config, indicator)),
    ]

    results: list[tuple[str, int]] = []
    print("Running quick functional check...")
    for name, runner in steps:
        print(f"\n== {name} ==")
        code = int(runner())
        results.append((name, code))
        print(f"{name} exit: {code}")
        if code != 0:
            break

    print("\nQuick check summary:")
    for name, code in results:
        print(f"- {name}: {'PASS' if code == 0 else 'FAIL'} (exit {code})")

    for _name, code in results:
        if code != 0:
            return code
    return 0


COMMAND_HANDLERS = {
    "pilot-readiness": command_pilot_readiness,
    "quick-check": command_quick_check,
    "capture": command_capture,
    "set-reference": command_set_reference,
    "inspect": command_inspect,
    "train": command_train,
    "production": command_production,
    "create-project": command_create_project,
    "switch-project": command_switch_project,
    "list-projects": command_list_projects,
    "project-manager": command_project_manager,
    "config-editor": command_config_editor,
    "dashboard": command_dashboard,
}


def main() -> int:
    config = load_config()
    # Default to dashboard/operator home if no mode is specified
    mode = sys.argv[1] if len(sys.argv) > 1 else "dashboard"
    handler = COMMAND_HANDLERS.get(mode)
    if handler is None:
        print_usage()
        return 2
    with IndicatorContext(config) as indicator:
        return handler(config, indicator, sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

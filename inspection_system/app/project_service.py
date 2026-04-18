#!/usr/bin/env python3
from inspection_system.app.camera_interface import create_project, get_current_project, list_projects, switch_project


def handle_create_project(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: python3 capture_test.py create-project <project_name> [description]")
        return 2

    project_name = argv[2]
    description = argv[3] if len(argv) > 3 else ""
    if create_project(project_name, description):
        print(f"Created project '{project_name}'")
        return 0

    print(f"Failed to create project '{project_name}'")
    return 1


def handle_switch_project(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: python3 capture_test.py switch-project <project_name>")
        return 2

    project_name = argv[2]
    if switch_project(project_name):
        print(f"Switched to project '{project_name}'")
        return 0

    print(f"Failed to switch to project '{project_name}'")
    return 1


def handle_list_projects() -> int:
    projects = list_projects()
    current = get_current_project()
    print(f"Current project: {current or 'None'}")
    print("Available projects:")
    for project in projects:
        status = " (ACTIVE)" if project["is_current"] else ""
        print(f"  {project['name']}{status}: {project['description']}")
    return 0

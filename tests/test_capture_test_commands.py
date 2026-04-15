from inspection_system.app.capture_test import COMMAND_HANDLERS


def test_quick_check_command_registered() -> None:
    assert "quick-check" in COMMAND_HANDLERS


def test_config_editor_command_registered() -> None:
    assert "config-editor" in COMMAND_HANDLERS

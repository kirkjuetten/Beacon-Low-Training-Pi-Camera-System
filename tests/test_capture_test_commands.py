from inspection_system.app.capture_test import COMMAND_HANDLERS


def test_quick_check_command_registered() -> None:
    assert "quick-check" in COMMAND_HANDLERS

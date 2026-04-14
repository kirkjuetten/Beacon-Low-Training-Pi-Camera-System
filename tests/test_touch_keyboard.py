from inspection_system.app.touch_keyboard import select_touch_keyboard_command


def _which_from_set(names: set[str]):
    def _which(name: str):
        return f"/usr/bin/{name}" if name in names else None

    return _which


def test_select_touch_keyboard_prefers_wvkbd_mobintl() -> None:
    cmd = select_touch_keyboard_command(_which_from_set({"wvkbd-mobintl", "matchbox-keyboard"}))
    assert cmd == ["wvkbd-mobintl"]


def test_select_touch_keyboard_falls_back_to_matchbox() -> None:
    cmd = select_touch_keyboard_command(_which_from_set({"matchbox-keyboard"}))
    assert cmd == ["matchbox-keyboard"]


def test_select_touch_keyboard_returns_none_when_missing() -> None:
    cmd = select_touch_keyboard_command(_which_from_set(set()))
    assert cmd is None

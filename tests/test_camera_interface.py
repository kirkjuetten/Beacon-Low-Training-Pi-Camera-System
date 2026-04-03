import tempfile
from pathlib import Path

from inspection_system.app import camera_interface


def test_build_capture_command_default() -> None:
    config = {
        "capture": {
            "timeout_ms": 250,
            "awb": "auto",
            "awb_gains": [1.2, 1.3],
            "shutter_us": 50000,
            "gain": 1.0,
            "width": 640,
            "height": 480,
            "rotation": 0,
            "hflip": True,
            "vflip": True,
        }
    }
    output_file = Path("/tmp/test.png")

    cmd = camera_interface.build_capture_command(config, output_file)

    assert cmd[0] == "rpicam-still"
    assert "--timeout" in cmd
    assert "--awb" in cmd
    assert "--awbgains" in cmd
    assert "--shutter" in cmd
    assert "--gain" in cmd
    assert "--width" in cmd
    assert "--height" in cmd
    assert "--rotation" in cmd
    assert "--hflip" in cmd
    assert "--vflip" in cmd
    assert "," in cmd[cmd.index("--awbgains") + 1]


def test_load_and_write_default_config_creates_file(monkeypatch, tmp_path) -> None:
    # Use a temporary config context to avoid touching user home
    monkeypatch.setattr(camera_interface, "BASE_DIR", tmp_path)
    monkeypatch.setattr(camera_interface, "APP_DIR", tmp_path / "app")
    monkeypatch.setattr(camera_interface, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(camera_interface, "REFERENCE_DIR", tmp_path / "reference")
    monkeypatch.setattr(camera_interface, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(camera_interface, "CONFIG_FILE", tmp_path / "config" / "camera_config.json")

    config = camera_interface.load_config()
    assert isinstance(config, dict)
    assert (tmp_path / "config" / "camera_config.json").exists()

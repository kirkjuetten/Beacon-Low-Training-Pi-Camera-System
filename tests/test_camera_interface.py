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


def test_get_active_runtime_paths_uses_current_project_registry(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(camera_interface, "BASE_DIR", tmp_path)
    monkeypatch.setattr(camera_interface, "APP_DIR", tmp_path / "app")
    monkeypatch.setattr(camera_interface, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(camera_interface, "REFERENCE_DIR", tmp_path / "reference")
    monkeypatch.setattr(camera_interface, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(camera_interface, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(camera_interface, "CONFIG_FILE", tmp_path / "config" / "camera_config.json")
    monkeypatch.setattr(camera_interface, "REFERENCE_MASK", tmp_path / "reference" / "golden_reference_mask.png")
    monkeypatch.setattr(camera_interface, "REFERENCE_IMAGE", tmp_path / "reference" / "golden_reference_image.png")

    camera_interface.ensure_directories()

    project_dir = tmp_path / "projects" / "widget_a"
    project_config = project_dir / "config" / "camera_config.json"
    project_reference_dir = project_dir / "reference"
    project_log_dir = project_dir / "logs"
    project_config.parent.mkdir(parents=True, exist_ok=True)
    project_reference_dir.mkdir(parents=True, exist_ok=True)
    project_log_dir.mkdir(parents=True, exist_ok=True)
    project_config.write_text("{}\n", encoding="utf-8")

    registry = {
        "current_project": "widget_a",
        "projects": {
            "widget_a": {
                "config_file": str(project_config),
                "reference_dir": str(project_reference_dir),
                "log_dir": str(project_log_dir),
            }
        },
    }
    (tmp_path / "config" / "projects.json").write_text(
        __import__("json").dumps(registry, indent=2) + "\n",
        encoding="utf-8",
    )

    paths = camera_interface.get_active_runtime_paths()

    assert paths["config_file"] == project_config
    assert paths["log_dir"] == project_log_dir
    assert paths["reference_mask"] == project_reference_dir / "golden_reference_mask.png"

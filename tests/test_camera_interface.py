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
    assert config["inspection"]["reference_strategy"] == "golden_only"
    assert config["inspection"]["blend_mode"] == "hard_only"
    assert config["inspection"]["tolerance_mode"] == "balanced"
    assert config["inspection"]["max_mean_edge_distance_px"] is None
    assert config["inspection"]["max_section_edge_distance_px"] is None
    assert config["inspection"]["max_section_width_delta_ratio"] is None
    assert config["inspection"]["max_section_center_offset_px"] is None


def test_load_config_merges_new_defaults_into_legacy_project_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(camera_interface, "BASE_DIR", tmp_path)
    monkeypatch.setattr(camera_interface, "APP_DIR", tmp_path / "app")
    monkeypatch.setattr(camera_interface, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(camera_interface, "REFERENCE_DIR", tmp_path / "reference")
    monkeypatch.setattr(camera_interface, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(camera_interface, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(camera_interface, "CONFIG_FILE", tmp_path / "config" / "camera_config.json")

    camera_interface.ensure_directories()

    project_dir = tmp_path / "projects" / "legacy"
    project_config = project_dir / "config" / "camera_config.json"
    project_config.parent.mkdir(parents=True, exist_ok=True)
    project_config.write_text(
        '{\n  "inspection": {\n    "inspection_mode": "mask_only"\n  }\n}\n',
        encoding="utf-8",
    )

    registry = {
        "current_project": "legacy",
        "projects": {
            "legacy": {
                "config_file": str(project_config),
                "reference_dir": str(project_dir / "reference"),
                "log_dir": str(project_dir / "logs"),
            }
        },
    }
    (tmp_path / "config" / "projects.json").write_text(
        __import__("json").dumps(registry, indent=2) + "\n",
        encoding="utf-8",
    )

    config = camera_interface.load_config()

    assert config["inspection"]["inspection_mode"] == "mask_only"
    assert config["inspection"]["reference_strategy"] == "golden_only"
    assert config["inspection"]["blend_mode"] == "hard_only"
    assert config["inspection"]["tolerance_mode"] == "balanced"
    assert config["inspection"]["max_mean_edge_distance_px"] is None
    assert config["inspection"]["max_section_edge_distance_px"] is None
    assert config["inspection"]["max_section_width_delta_ratio"] is None
    assert config["inspection"]["max_section_center_offset_px"] is None


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


def test_clone_project_copies_project_files_and_registry(monkeypatch, tmp_path) -> None:
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
    assert camera_interface.create_project("alpha", "source project") is True

    source_ref = tmp_path / "projects" / "alpha" / "reference"
    marker = source_ref / "sample.png"
    marker.write_bytes(b"img")

    assert camera_interface.clone_project("alpha", "alpha_clone") is True

    cloned_marker = tmp_path / "projects" / "alpha_clone" / "reference" / "sample.png"
    assert cloned_marker.exists()

    registry = camera_interface.get_project_registry()
    assert "alpha_clone" in registry["projects"]
    assert registry["projects"]["alpha_clone"]["description"] == "source project"


def test_rename_project_updates_registry_and_current_project(monkeypatch, tmp_path) -> None:
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
    assert camera_interface.create_project("beta", "rename me") is True
    assert camera_interface.switch_project("beta") is True

    assert camera_interface.rename_project("beta", "beta_renamed") is True

    registry = camera_interface.get_project_registry()
    assert "beta" not in registry["projects"]
    assert "beta_renamed" in registry["projects"]
    assert registry.get("current_project") == "beta_renamed"

    assert (tmp_path / "projects" / "beta").exists() is False
    assert (tmp_path / "projects" / "beta_renamed").exists() is True


def test_delete_active_project_switches_to_remaining_project(monkeypatch, tmp_path) -> None:
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
    assert camera_interface.create_project("alpha", "first") is True
    assert camera_interface.create_project("beta", "second") is True
    assert camera_interface.switch_project("beta") is True

    assert camera_interface.delete_project("beta") is True

    registry = camera_interface.get_project_registry()
    assert "beta" not in registry["projects"]
    assert registry["current_project"] == "alpha"
    assert camera_interface.CONFIG_FILE == tmp_path / "projects" / "alpha" / "config" / "camera_config.json"
    assert (tmp_path / "projects" / "beta").exists() is False


def test_delete_last_active_project_clears_current_project(monkeypatch, tmp_path) -> None:
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
    assert camera_interface.create_project("solo", "only") is True
    assert camera_interface.switch_project("solo") is True

    assert camera_interface.delete_project("solo") is True

    registry = camera_interface.get_project_registry()
    assert registry["projects"] == {}
    assert registry["current_project"] is None
    assert camera_interface.CONFIG_FILE == tmp_path / "config" / "camera_config.json"
    assert (tmp_path / "projects" / "solo").exists() is False


def test_delete_project_accepts_whitespace_padded_name(monkeypatch, tmp_path) -> None:
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
    assert camera_interface.create_project("5036", "pad print") is True

    assert camera_interface.delete_project(" 5036 ") is True

    registry = camera_interface.get_project_registry()
    assert "5036" not in registry["projects"]
    assert (tmp_path / "projects" / "5036").exists() is False


def test_delete_project_uses_registry_config_path_to_find_project_dir(monkeypatch, tmp_path) -> None:
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

    actual_project_dir = tmp_path / "projects" / "5036"
    (actual_project_dir / "config").mkdir(parents=True, exist_ok=True)
    (actual_project_dir / "reference").mkdir(parents=True, exist_ok=True)
    (actual_project_dir / "logs").mkdir(parents=True, exist_ok=True)
    (actual_project_dir / "config" / "camera_config.json").write_text("{}\n", encoding="utf-8")

    registry = {
        "current_project": None,
        "projects": {
            "5036": {
                "description": "pad print",
                "created": "2026-04-16 12:00:00",
                "config_file": str(actual_project_dir / "config" / "camera_config.json"),
                "reference_dir": str(actual_project_dir / "reference"),
                "log_dir": str(actual_project_dir / "logs"),
            }
        },
    }
    (tmp_path / "config" / "projects.json").write_text(
        __import__("json").dumps(registry, indent=2) + "\n",
        encoding="utf-8",
    )

    assert camera_interface.delete_project("5036") is True
    assert actual_project_dir.exists() is False

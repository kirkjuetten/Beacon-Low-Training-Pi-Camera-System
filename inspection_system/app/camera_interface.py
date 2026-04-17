#!/usr/bin/env python3
import json
import time
from pathlib import Path
from typing import Optional, Dict, List

BASE_DIR = Path.home() / "inspection_system"
APP_DIR = BASE_DIR / "app"
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = BASE_DIR / "logs"
REFERENCE_DIR = BASE_DIR / "reference"
PROJECTS_DIR = BASE_DIR / "projects"
CONFIG_FILE = CONFIG_DIR / "camera_config.json"
PROJECT_REGISTRY_FILE = CONFIG_DIR / "projects.json"
REFERENCE_MASK = REFERENCE_DIR / "golden_reference_mask.png"
REFERENCE_IMAGE = REFERENCE_DIR / "golden_reference_image.png"
TEMP_IMAGE = BASE_DIR / "temp_capture.png"


def _reset_global_runtime_paths() -> None:
    global CONFIG_FILE, REFERENCE_DIR, LOG_DIR, REFERENCE_MASK, REFERENCE_IMAGE
    CONFIG_FILE = CONFIG_DIR / "camera_config.json"
    REFERENCE_DIR = BASE_DIR / "reference"
    LOG_DIR = BASE_DIR / "logs"
    REFERENCE_MASK = REFERENCE_DIR / "golden_reference_mask.png"
    REFERENCE_IMAGE = REFERENCE_DIR / "golden_reference_image.png"


def _set_global_runtime_paths(project_info: Dict) -> None:
    global CONFIG_FILE, REFERENCE_DIR, LOG_DIR, REFERENCE_MASK, REFERENCE_IMAGE
    CONFIG_FILE = Path(project_info["config_file"])
    REFERENCE_DIR = Path(project_info["reference_dir"])
    LOG_DIR = Path(project_info["log_dir"])
    REFERENCE_MASK = REFERENCE_DIR / "golden_reference_mask.png"
    REFERENCE_IMAGE = REFERENCE_DIR / "golden_reference_image.png"


def _resolve_project_registry_key(registry: Dict, project_name: str) -> Optional[str]:
    normalized_name = str(project_name).strip()
    if normalized_name in registry["projects"]:
        return normalized_name
    for existing_name in registry["projects"]:
        if str(existing_name).strip() == normalized_name:
            return existing_name
    return None


def _deep_merge_defaults(defaults, config):
    if isinstance(defaults, dict) and isinstance(config, dict):
        merged = {}
        for key, value in defaults.items():
            if key in config:
                merged[key] = _deep_merge_defaults(value, config[key])
            else:
                merged[key] = value
        for key, value in config.items():
            if key not in merged:
                merged[key] = value
        return merged
    return config

DEFAULT_CONFIG = {
    "capture": {
        "timeout_ms": 200,
        "awb": None,
        "awb_gains": [1.8, 1.8],
        "shutter_us": 40000,
        "gain": 0.5,
        "width": None,
        "height": None,
        "rotation": 0,
        "hflip": False,
        "vflip": False,
    },
    "inspection": {
        "enabled": True,
        "inspection_mode": "mask_only",
        "reference_strategy": "golden_only",
        "blend_mode": "hard_only",
        "tolerance_mode": "balanced",
        "roi": {
            "x": 0,
            "y": 0,
            "width": None,
            "height": None,
        },
        "threshold_mode": "otsu",
        "threshold_value": 180,
        "blur_kernel": 3,
        "reference_erode_iterations": 1,
        "reference_dilate_iterations": 1,
        "sample_erode_iterations": 1,
        "sample_dilate_iterations": 1,
        "min_feature_pixels": 100,
        "save_debug_images": True,
        "allowed_dilate_iterations": 2,
        "required_erode_iterations": 1,
        "max_outside_allowed_ratio": 0.02,
        "min_required_coverage": 0.92,
        "min_section_coverage": 0.85,
        "max_mean_edge_distance_px": None,
        "max_section_edge_distance_px": None,
        "max_section_width_delta_ratio": None,
        "max_section_center_offset_px": None,
        "section_columns": 12,
    },
    "alignment": {
        "enabled": True,
        "mode": "moments",
        "tolerance_profile": "balanced",
        "max_angle_deg": 1.0,
        "max_shift_x": 4,
        "max_shift_y": 3,
    },
    "indicator_led": {
        "enabled": False,
        "pass_gpio": 23,
        "fail_gpio": 24,
        "pulse_ms": 750,
    },
}


def ensure_directories() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def write_default_config() -> None:
    CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")


def load_config() -> dict:
    ensure_directories()

    # If we have a current project, use its config
    current_project = get_current_project()
    if current_project:
        registry = get_project_registry()
        project_info = registry["projects"].get(current_project)
        if project_info:
            project_config_file = Path(project_info["config_file"])
            if project_config_file.exists():
                with project_config_file.open("r", encoding="utf-8") as f:
                    return _deep_merge_defaults(DEFAULT_CONFIG, json.load(f))

    # Fall back to global config
    if not CONFIG_FILE.exists():
        write_default_config()
        print(f"Created default config: {CONFIG_FILE}")

    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return _deep_merge_defaults(DEFAULT_CONFIG, json.load(f))


def import_cv2_and_numpy():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        return cv2, np
    except ImportError as exc:
        print("OpenCV and NumPy are required for inspection mode.")
        print("Install them with:")
        print("sudo apt install -y python3-opencv python3-numpy")
        raise SystemExit(2) from exc


# Project Management Functions

def get_project_registry_path() -> Path:
    return CONFIG_DIR / "projects.json"


def get_project_registry() -> Dict:
    """Load the project registry."""
    ensure_directories()
    project_registry_file = get_project_registry_path()

    if not project_registry_file.exists():
        # Create default registry
        registry = {
            "current_project": None,
            "projects": {}
        }
        project_registry_file.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        return registry

    with project_registry_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_project_registry(registry: Dict) -> None:
    """Save the project registry."""
    project_registry_file = get_project_registry_path()
    project_registry_file.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def create_project(project_name: str, description: str = "") -> bool:
    """Create a new project with its own config and reference files."""
    registry = get_project_registry()

    if project_name in registry["projects"]:
        print(f"Project '{project_name}' already exists.")
        return False

    # Create project directory structure
    project_dir = PROJECTS_DIR / project_name
    project_config_dir = project_dir / "config"
    project_reference_dir = project_dir / "reference"
    project_log_dir = project_dir / "logs"

    project_config_dir.mkdir(parents=True, exist_ok=True)
    project_reference_dir.mkdir(parents=True, exist_ok=True)
    project_log_dir.mkdir(parents=True, exist_ok=True)

    # Copy default config to project
    project_config_file = project_config_dir / "camera_config.json"
    if not project_config_file.exists():
        project_config_file.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")

    # Add to registry
    registry["projects"][project_name] = {
        "description": description,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_file": str(project_config_file),
        "reference_dir": str(project_reference_dir),
        "log_dir": str(project_log_dir)
    }

    save_project_registry(registry)
    print(f"Created project '{project_name}'")
    return True


def switch_project(project_name: str) -> bool:
    """Switch to a different project."""
    registry = get_project_registry()

    if project_name not in registry["projects"]:
        print(f"Project '{project_name}' does not exist.")
        return False

    registry["current_project"] = project_name
    save_project_registry(registry)

    project_info = registry["projects"][project_name]
    _set_global_runtime_paths(project_info)

    print(f"Switched to project '{project_name}'")
    return True


def get_current_project() -> Optional[str]:
    """Get the name of the currently active project."""
    registry = get_project_registry()
    return registry.get("current_project")


def get_active_runtime_paths() -> Dict[str, Path]:
    """Resolve config, log, and reference paths for the active project context."""
    current_project = get_current_project()
    if current_project:
        registry = get_project_registry()
        project_info = registry["projects"].get(current_project)
        if project_info:
            reference_dir = Path(project_info["reference_dir"])
            return {
                "config_file": Path(project_info["config_file"]),
                "log_dir": Path(project_info["log_dir"]),
                "reference_dir": reference_dir,
                "reference_mask": reference_dir / "golden_reference_mask.png",
                "reference_image": reference_dir / "golden_reference_image.png",
            }

    return {
        "config_file": CONFIG_FILE,
        "log_dir": LOG_DIR,
        "reference_dir": REFERENCE_DIR,
        "reference_mask": REFERENCE_MASK,
        "reference_image": REFERENCE_IMAGE,
    }


def list_projects() -> List[Dict]:
    """List all available projects."""
    registry = get_project_registry()
    projects = []
    for name, info in registry["projects"].items():
        projects.append({
            "name": name,
            "description": info.get("description", ""),
            "created": info.get("created", ""),
            "is_current": name == registry.get("current_project")
        })
    return projects


def delete_project(project_name: str) -> bool:
    """Delete a project and all its files."""
    registry = get_project_registry()
    resolved_project_name = _resolve_project_registry_key(registry, project_name)

    if resolved_project_name is None:
        print(f"Project '{str(project_name).strip()}' does not exist.")
        return False

    deleting_current = registry.get("current_project") == resolved_project_name

    # Remove project directory
    project_info = registry["projects"][resolved_project_name]
    project_dir = Path(project_info.get("config_file", PROJECTS_DIR / resolved_project_name)).parent.parent
    if project_dir.exists():
        import shutil
        try:
            shutil.rmtree(project_dir)
        except Exception as exc:
            print(f"Failed to delete project '{resolved_project_name}': {exc}")
            return False

    # Remove from registry
    del registry["projects"][resolved_project_name]

    if deleting_current:
        remaining_projects = sorted(registry["projects"])
        if remaining_projects:
            replacement_project = remaining_projects[0]
            registry["current_project"] = replacement_project
            _set_global_runtime_paths(registry["projects"][replacement_project])
        else:
            registry["current_project"] = None
            _reset_global_runtime_paths()

    save_project_registry(registry)

    print(f"Deleted project '{resolved_project_name}'")
    return True


def clone_project(source_project: str, new_project: str, description: Optional[str] = None) -> bool:
    """Clone an existing project into a new project name."""
    registry = get_project_registry()

    if source_project not in registry["projects"]:
        print(f"Project '{source_project}' does not exist.")
        return False

    if new_project in registry["projects"]:
        print(f"Project '{new_project}' already exists.")
        return False

    source_dir = PROJECTS_DIR / source_project
    if not source_dir.exists():
        print(f"Source project directory does not exist: {source_dir}")
        return False

    target_dir = PROJECTS_DIR / new_project

    try:
        import shutil

        shutil.copytree(source_dir, target_dir)
    except Exception as exc:
        print(f"Failed to clone project '{source_project}': {exc}")
        return False

    source_info = registry["projects"][source_project]
    registry["projects"][new_project] = {
        "description": source_info.get("description", "") if description is None else description,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_file": str(target_dir / "config" / "camera_config.json"),
        "reference_dir": str(target_dir / "reference"),
        "log_dir": str(target_dir / "logs"),
    }
    save_project_registry(registry)
    print(f"Cloned project '{source_project}' to '{new_project}'")
    return True


def rename_project(old_name: str, new_name: str) -> bool:
    """Rename an existing project and keep its data and active selection."""
    registry = get_project_registry()

    if old_name not in registry["projects"]:
        print(f"Project '{old_name}' does not exist.")
        return False

    if new_name in registry["projects"]:
        print(f"Project '{new_name}' already exists.")
        return False

    old_dir = PROJECTS_DIR / old_name
    new_dir = PROJECTS_DIR / new_name

    if not old_dir.exists():
        print(f"Project directory does not exist: {old_dir}")
        return False

    try:
        import shutil

        shutil.move(str(old_dir), str(new_dir))
    except Exception as exc:
        print(f"Failed to rename project '{old_name}': {exc}")
        return False

    old_info = registry["projects"].pop(old_name)
    registry["projects"][new_name] = {
        "description": old_info.get("description", ""),
        "created": old_info.get("created", time.strftime("%Y-%m-%d %H:%M:%S")),
        "config_file": str(new_dir / "config" / "camera_config.json"),
        "reference_dir": str(new_dir / "reference"),
        "log_dir": str(new_dir / "logs"),
    }

    if registry.get("current_project") == old_name:
        registry["current_project"] = new_name
        _set_global_runtime_paths(registry["projects"][new_name])

    save_project_registry(registry)
    print(f"Renamed project '{old_name}' to '{new_name}'")
    return True


def export_project(project_name: str, export_path: Path) -> bool:
    """Export a project to a zip file."""
    registry = get_project_registry()

    if project_name not in registry["projects"]:
        print(f"Project '{project_name}' does not exist.")
        return False

    project_dir = PROJECTS_DIR / project_name
    if not project_dir.exists():
        print(f"Project directory does not exist: {project_dir}")
        return False

    try:
        import shutil
        shutil.make_archive(str(export_path), 'zip', project_dir)
        print(f"Exported project '{project_name}' to {export_path}.zip")
        return True
    except Exception as e:
        print(f"Failed to export project: {e}")
        return False


def import_project(zip_path: Path, project_name: Optional[str] = None) -> bool:
    """Import a project from a zip file."""
    if not zip_path.exists():
        print(f"Zip file does not exist: {zip_path}")
        return False

    if project_name is None:
        project_name = zip_path.stem

    registry = get_project_registry()
    if project_name in registry["projects"]:
        print(f"Project '{project_name}' already exists.")
        return False

    try:
        import shutil
        project_dir = PROJECTS_DIR / project_name

        # Extract zip
        shutil.unpack_archive(str(zip_path), str(project_dir))

        # Validate project structure
        config_dir = project_dir / "config"
        reference_dir = project_dir / "reference"
        log_dir = project_dir / "logs"

        if not config_dir.exists() or not reference_dir.exists():
            print("Invalid project structure in zip file.")
            shutil.rmtree(project_dir)
            return False

        # Add to registry
        registry["projects"][project_name] = {
            "description": f"Imported from {zip_path.name}",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_file": str(config_dir / "camera_config.json"),
            "reference_dir": str(reference_dir),
            "log_dir": str(log_dir)
        }

        save_project_registry(registry)
        print(f"Imported project '{project_name}'")
        return True

    except Exception as e:
        print(f"Failed to import project: {e}")
        return False


def build_capture_command(config: dict, output_file: Path) -> list[str]:
    capture = config.get("capture", {})
    cmd = [
        "rpicam-still",
        "--nopreview",
        "-o",
        str(output_file),
        "--timeout",
        str(capture.get("timeout_ms", 200)),
    ]

    awb = capture.get("awb")
    if awb:
        cmd.extend(["--awb", str(awb)])

    awb_gains = capture.get("awb_gains")
    if isinstance(awb_gains, (list, tuple)) and len(awb_gains) == 2:
        cmd.extend(["--awbgains", f"{awb_gains[0]},{awb_gains[1]}"])

    shutter_us = capture.get("shutter_us")
    if shutter_us is not None:
        cmd.extend(["--shutter", str(shutter_us)])

    gain = capture.get("gain")
    if gain is not None:
        cmd.extend(["--gain", str(gain)])

    width = capture.get("width")
    height = capture.get("height")
    if width and height:
        cmd.extend(["--width", str(width), "--height", str(height)])

    rotation = capture.get("rotation", 0)
    if rotation in (0, 180):
        cmd.extend(["--rotation", str(rotation)])

    if capture.get("hflip", False):
        cmd.append("--hflip")

    if capture.get("vflip", False):
        cmd.append("--vflip")

    return cmd


class IndicatorLED:
    def __init__(self, enabled: bool, pass_gpio: int, fail_gpio: int, pulse_ms: int) -> None:
        self.enabled = enabled
        self.pass_gpio = pass_gpio
        self.fail_gpio = fail_gpio
        self.pulse_ms = pulse_ms
        self.gpio = None

        if not self.enabled:
            return

        try:
            import RPi.GPIO as GPIO  # type: ignore
        except ImportError:
            print("RPi.GPIO is not installed. LED output disabled.")
            self.enabled = False
            return

        self.gpio = GPIO
        self.gpio.setwarnings(False)
        self.gpio.setmode(GPIO.BCM)
        try:
            self.gpio.setup(self.pass_gpio, GPIO.OUT, initial=GPIO.LOW)
            self.gpio.setup(self.fail_gpio, GPIO.OUT, initial=GPIO.LOW)
        except RuntimeError as e:
            print(f"GPIO setup failed: {e}. LED output disabled.")
            self.enabled = False
            return

    def pulse_pass(self) -> None:
        self._pulse(self.pass_gpio)

    def pulse_fail(self) -> None:
        self._pulse(self.fail_gpio)

    def _pulse(self, pin: int) -> None:
        if not self.enabled or self.gpio is None:
            return

        self.gpio.output(self.pass_gpio, self.gpio.LOW)
        self.gpio.output(self.fail_gpio, self.gpio.LOW)
        self.gpio.output(pin, self.gpio.HIGH)
        time.sleep(self.pulse_ms / 1000.0)
        self.gpio.output(pin, self.gpio.LOW)

    def cleanup(self) -> None:
        if self.enabled and self.gpio is not None:
            self.gpio.cleanup((self.pass_gpio, self.fail_gpio))

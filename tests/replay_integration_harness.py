from __future__ import annotations

import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "5096v2.0"
SESSION_RELATIVE_PATH = Path("test_data") / "5096v2-0" / "5096v2-0-default" / "20260417_152838_commissioning"
REAL_PROJECT_ROOT = REPO_ROOT / "inspection_system" / "projects" / PROJECT_ID
REAL_SESSION_DIR = REAL_PROJECT_ROOT / SESSION_RELATIVE_PATH


def _load_session_records(session_dir: Path) -> list[dict]:
    manifest_path = session_dir / "captures.jsonl"
    return [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_replay_project_fixture(
    tmp_path: Path,
    *,
    selected_images: tuple[str, ...],
    split_overrides: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    if not REAL_PROJECT_ROOT.exists():
        raise FileNotFoundError(f"Missing committed replay project snapshot: {REAL_PROJECT_ROOT}")
    if not REAL_SESSION_DIR.exists():
        raise FileNotFoundError(f"Missing committed replay session snapshot: {REAL_SESSION_DIR}")

    project_root = tmp_path / "inspection_system" / "projects" / PROJECT_ID
    config_dir = project_root / "config"
    reference_dir = project_root / "reference"
    logs_dir = project_root / "logs"
    session_dir = project_root / SESSION_RELATIVE_PATH
    images_dir = session_dir / "images"

    config_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(REAL_PROJECT_ROOT / "config" / "camera_config.json", config_dir / "camera_config.json")
    for child in (REAL_PROJECT_ROOT / "reference").iterdir():
        target = reference_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)

    config_path = config_dir / "camera_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inspection_cfg = config.setdefault("inspection", {})
    inspection_cfg["save_debug_images"] = False
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    split_overrides = split_overrides or {}
    selected = set(selected_images)
    records_by_name = {
        Path(str(record.get("relative_image_path") or record.get("image_path", ""))).name: record
        for record in _load_session_records(REAL_SESSION_DIR)
    }
    missing_images = sorted(selected.difference(records_by_name))
    if missing_images:
        raise FileNotFoundError(f"Missing replay fixture records for images: {', '.join(missing_images)}")

    subset_records: list[dict] = []
    for image_name in selected_images:
        source_record = dict(records_by_name[image_name])
        shutil.copy2(REAL_SESSION_DIR / "images" / image_name, images_dir / image_name)
        source_record["relative_image_path"] = f"images/{image_name}"
        source_record["image_path"] = str(images_dir / image_name)
        if image_name in split_overrides:
            source_record["dataset_split"] = split_overrides[image_name]
        subset_records.append(source_record)

    (session_dir / "captures.jsonl").write_text(
        "\n".join(json.dumps(record) for record in subset_records) + "\n",
        encoding="utf-8",
    )

    session_json = REAL_SESSION_DIR / "session.json"
    if session_json.exists():
        shutil.copy2(session_json, session_dir / "session.json")

    return project_root, session_dir
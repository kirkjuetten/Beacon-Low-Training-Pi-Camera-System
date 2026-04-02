#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from capture_test import load_config
from replay_inspection import VALID_SUFFIXES, inspect_file
from replay_summary_utils import summarize_results
from result_status import CONFIG_ERROR, FAIL, INVALID_CAPTURE


def inspect_folder_with_summary(config: dict, folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        print(json.dumps({"status": CONFIG_ERROR, "reason": f"Folder not found: {folder}"}))
        return 2

    image_paths = sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )

    if not image_paths:
        print(json.dumps({"status": CONFIG_ERROR, "reason": f"No images found in: {folder}"}))
        return 2

    results: list[dict] = []
    exit_code = 0
    for image_path in image_paths:
        result = inspect_file(config, image_path)
        results.append(result)
        print(json.dumps(result, sort_keys=True))
        if result["status"] in {FAIL, INVALID_CAPTURE, CONFIG_ERROR}:
            exit_code = 1

    print(json.dumps(summarize_results(results), sort_keys=True))
    return exit_code


def print_usage() -> None:
    print("Usage:")
    print("  python3 inspection_system/app/replay_summary.py <folder_path>")


def main() -> int:
    if len(sys.argv) != 2:
        print_usage()
        return 2

    folder = Path(sys.argv[1])
    config = load_config()
    return inspect_folder_with_summary(config, folder)


if __name__ == "__main__":
    raise SystemExit(main())
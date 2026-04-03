#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path

from inspection_system.app.camera_interface import build_capture_command, TEMP_IMAGE


def cleanup_temp_image() -> None:
    if TEMP_IMAGE.exists():
        TEMP_IMAGE.unlink()


def capture_to_temp(config: dict) -> tuple[int, Path, str]:
    cleanup_temp_image()

    # Simple file locking to prevent concurrent access
    lock_file = TEMP_IMAGE.with_suffix('.lock')
    max_wait = 10  # seconds
    wait_time = 0.1

    start_time = time.time()
    while lock_file.exists() and (time.time() - start_time) < max_wait:
        time.sleep(wait_time)

    if lock_file.exists():
        return 1, TEMP_IMAGE, "Temp file locked by another process"

    try:
        # Create lock file
        lock_file.write_text("locked")

        cmd = build_capture_command(config, TEMP_IMAGE)
        print("Capturing image...")
        print("Command:", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
        except subprocess.TimeoutExpired:
            print("Camera capture timed out after 30 seconds")
            return 1, TEMP_IMAGE, "Camera capture timeout"
        stderr_text = (result.stderr or "").strip()
        return result.returncode, TEMP_IMAGE, stderr_text
    finally:
        # Clean up lock file
        if lock_file.exists():
            lock_file.unlink()

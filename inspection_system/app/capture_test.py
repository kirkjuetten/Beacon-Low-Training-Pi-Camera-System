#!/usr/bin/env python3
import json
import subprocess
import sys
import time
from pathlib import Path

from alignment_utils import align_sample_mask
from inspection_pipeline import inspect_against_reference
from morphology_utils import dilate_mask, erode_mask
from preprocessing_utils import get_roi, make_binary_mask
from reference_region_utils import build_reference_regions
from scoring_utils import evaluate_metrics, score_sample
from section_mask_utils import compute_section_masks

BASE_DIR = Path.home() / "inspection_system"
APP_DIR = BASE_DIR / "app"
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = BASE_DIR / "logs"
REFERENCE_DIR = BASE_DIR / "reference"
CONFIG_FILE = CONFIG_DIR / "camera_config.json"
REFERENCE_MASK = REFERENCE_DIR / "golden_reference_mask.png"
REFERENCE_IMAGE = REFERENCE_DIR / "golden_reference_image.png"

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
        "vflip": False
    },
    "inspection": {
        "enabled": True,
        "roi": {
            "x": 460,
            "y": 360,
            "width": 680,
            "height": 220
        },
        "threshold_mode": "otsu",
        "threshold_value": 180,
        "blur_kernel": 3,
        "reference_erode_iterations": 1,
        "reference_dilate_iterations": 1,
        "sample_erode_iterations": 1,
        "sample_dilate_iterations": 1,
        "min_white_pixels": 100,
        "save_debug_images": True,
        "allowed_dilate_iterations": 2,
        "required_erode_iterations": 1,
        "max_outside_allowed_ratio": 0.02,
        "min_required_coverage": 0.92,
        "min_section_coverage": 0.85,
        "section_columns": 12
    },
    "alignment": {
        "enabled": True,
        "mode": "moments",
        "max_angle_deg": 1.0,
        "max_shift_x": 4,
        "max_shift_y": 3
    },
    "indicator_led": {
        "enabled": False,
        "pass_gpio": 23,
        "fail_gpio": 24,
        "pulse_ms": 750
    }
}


def ensure_directories() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


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
        self.gpio.setup(self.pass_gpio, GPIO.OUT, initial=GPIO.LOW)
        self.gpio.setup(self.fail_gpio, GPIO.OUT, initial=GPIO.LOW)

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


def write_default_config() -> None:
    CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")


def load_config() -> dict:
    ensure_directories()
    if not CONFIG_FILE.exists():
        write_default_config()
        print(f"Created default config: {CONFIG_FILE}")

    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def build_capture_command(config: dict, output_file: Path) -> list[str]:
    capture = config.get("capture", {})
    cmd = [
        "rpicam-still",
        "-o", str(output_file),
        "--timeout", str(capture.get("timeout_ms", 200)),
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


def cleanup_temp_image() -> None:
    if TEMP_IMAGE.exists():
        TEMP_IMAGE.unlink()


def capture_to_temp(config: dict) -> tuple[int, Path, str]:
    cleanup_temp_image()
    cmd = build_capture_command(config, TEMP_IMAGE)
    print("Capturing image...")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr_text = (result.stderr or "").strip()
    return result.returncode, TEMP_IMAGE, stderr_text




def save_debug_outputs(stem: str, aligned_sample_mask, diff_image) -> dict:
    cv2, _ = import_cv2_and_numpy()
    debug_mask_path = LOG_DIR / f"{stem}_mask.png"
    debug_diff_path = LOG_DIR / f"{stem}_diff.png"
    cv2.imwrite(str(debug_mask_path), aligned_sample_mask)
    cv2.imwrite(str(debug_diff_path), diff_image)
    return {
        "mask": str(debug_mask_path),
        "diff": str(debug_diff_path),
    }


def set_reference(config: dict) -> int:
    result_code, image_path, stderr_text = capture_to_temp(config)
    if result_code != 0:
        print("Reference capture failed.")
        if stderr_text:
            print(stderr_text)
        cleanup_temp_image()
        return result_code

    try:
        cv2, np = import_cv2_and_numpy()
        inspection_cfg = config.get("inspection", {})
        roi_image, _, mask, _, _, _ = make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy)
        reference_erode_iterations = int(inspection_cfg.get("reference_erode_iterations", 1))
        reference_dilate_iterations = int(inspection_cfg.get("reference_dilate_iterations", 1))
        mask = erode_mask(mask, reference_erode_iterations, cv2, np)
        mask = dilate_mask(mask, reference_dilate_iterations, cv2, np)

        white_pixels = int((mask > 0).sum())
        min_white_pixels = int(inspection_cfg.get("min_white_pixels", 100))
        if white_pixels < min_white_pixels:
            print("Reference mask did not produce enough white pixels.")
            print(f"White pixels found: {white_pixels}")
            print("Adjust ROI / threshold before using this reference.")
            return 3

        cv2.imwrite(str(REFERENCE_MASK), mask)
        cv2.imwrite(str(REFERENCE_IMAGE), roi_image)
        print(f"Saved reference mask: {REFERENCE_MASK}")
        print(f"Saved reference image: {REFERENCE_IMAGE}")
        print(f"Reference white pixels: {white_pixels}")
        return 0
    finally:
        cleanup_temp_image()


def print_inspection_result(passed: bool, details: dict) -> None:
    print("Inspection result:", "PASS" if passed else "FAIL")
    print(f"ROI: {details['roi']}")
    print(f"Best angle correction: {details.get('best_angle_deg', 0.0):.2f} deg")
    print(f"Best shift correction: x={details.get('best_shift_x', 0)}, y={details.get('best_shift_y', 0)} px")
    print(f"Required coverage: {details['required_coverage']:.4f} (min {details['min_required_coverage']:.4f})")
    print(f"Outside allowed ratio: {details['outside_allowed_ratio']:.4f} (max {details['max_outside_allowed_ratio']:.4f})")
    print(f"Min section coverage: {details['min_section_coverage']:.4f} (min {details['min_section_coverage_limit']:.4f})")
    print(f"Sample white pixels: {details['sample_white_pixels']}")
    if details.get("section_coverages"):
        print("Section coverages:", ", ".join(f"{v:.3f}" for v in details["section_coverages"]))
    if "ssim" in details:
        print(f"SSIM: {details['ssim']:.4f}")
    if "histogram_similarity" in details:
        print(f"Histogram similarity: {details['histogram_similarity']:.4f}")
    if "mse" in details:
        print(f"MSE: {details['mse']:.2f}")
    if "anomaly_score" in details:
        print(f"Anomaly score: {details['anomaly_score']:.4f}")
    if details.get("debug_paths"):
        for key, path in details["debug_paths"].items():
            print(f"Debug {key}: {path}")


def run_capture_only(config: dict) -> int:
    result_code, image_path, stderr_text = capture_to_temp(config)
    if result_code != 0:
        print("Capture failed.")
        if stderr_text:
            print(stderr_text)
        cleanup_temp_image()
        return result_code

    print("Temporary capture completed.")
    cleanup_temp_image()
    return 0


def run_capture_and_inspect(config: dict, indicator: IndicatorLED) -> int:
    result_code, image_path, stderr_text = capture_to_temp(config)
    if result_code != 0:
        print("Capture failed.")
        if stderr_text:
            print(stderr_text)
        indicator.pulse_fail()
        cleanup_temp_image()
        return result_code

    try:
        passed, details = inspect_against_reference(
            config,
            image_path,
            make_binary_mask,
            REFERENCE_MASK,
            REFERENCE_IMAGE,
            align_sample_mask,
            build_reference_regions,
            compute_section_masks,
            score_sample,
            evaluate_metrics,
            save_debug_outputs,
            import_cv2_and_numpy,
            dilate_mask,
            erode_mask,
            anomaly_detector=None,
        )
        print_inspection_result(passed, details)
        if passed:
            indicator.pulse_pass()
            return 0

        indicator.pulse_fail()
        return 1
    finally:
        cleanup_temp_image()


def main() -> int:
    config = load_config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"

    led_cfg = config.get("indicator_led", {})
    indicator = IndicatorLED(
        enabled=bool(led_cfg.get("enabled", False)),
        pass_gpio=int(led_cfg.get("pass_gpio", 23)),
        fail_gpio=int(led_cfg.get("fail_gpio", 24)),
        pulse_ms=int(led_cfg.get("pulse_ms", 750)),
    )

    try:
        if mode == "capture":
            return run_capture_only(config)
        if mode == "set-reference":
            return set_reference(config)
        if mode == "inspect":
            return run_capture_and_inspect(config, indicator)

        print("Usage:")
        print("  python3 capture_test.py capture")
        print("  python3 capture_test.py set-reference")
        print("  python3 capture_test.py inspect")
        return 2
    finally:
        indicator.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

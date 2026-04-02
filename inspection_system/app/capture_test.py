#!/usr/bin/env python3
import json
import subprocess
import sys
import time
from pathlib import Path

from alignment_utils import align_sample_mask
from morphology_utils import dilate_mask, erode_mask
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
TEMP_IMAGE = BASE_DIR / "temp_capture.jpg"

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


def get_roi(image, roi_cfg: dict):
    x = int(roi_cfg.get("x", 0))
    y = int(roi_cfg.get("y", 0))
    w = int(roi_cfg.get("width", 0))
    h = int(roi_cfg.get("height", 0))

    if w <= 0 or h <= 0:
        return image, (0, 0, image.shape[1], image.shape[0])

    x2 = min(x + w, image.shape[1])
    y2 = min(y + h, image.shape[0])
    x = max(0, x)
    y = max(0, y)

    if x >= x2 or y >= y2:
        raise ValueError("Configured ROI is outside the image bounds.")

    return image[y:y2, x:x2], (x, y, x2 - x, y2 - y)


def make_binary_mask(image_path: Path, config: dict):
    cv2, np = import_cv2_and_numpy()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    inspection = config.get("inspection", {})
    roi_image, roi = get_roi(image, inspection.get("roi", {}))
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    blur_kernel = int(inspection.get("blur_kernel", 3))
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    threshold_mode = str(inspection.get("threshold_mode", "fixed")).lower()
    threshold_value = int(inspection.get("threshold_value", 180))

    if threshold_mode == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    return roi_image, gray, mask, roi, cv2, np






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
        _, _, mask, _, _, _ = make_binary_mask(image_path, config)
        inspection_cfg = config.get("inspection", {})
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

        cv2, _ = import_cv2_and_numpy()
        cv2.imwrite(str(REFERENCE_MASK), mask)
        print(f"Saved reference mask: {REFERENCE_MASK}")
        print(f"Reference white pixels: {white_pixels}")
        return 0
    finally:
        cleanup_temp_image()


def inspect_against_reference(config: dict, image_path: Path) -> tuple[bool, dict]:
    _, _, sample_mask, roi, cv2, np = make_binary_mask(image_path, config)
    inspection_cfg = config.get("inspection", {})

    sample_erode_iterations = int(inspection_cfg.get("sample_erode_iterations", 1))
    sample_dilate_iterations = int(inspection_cfg.get("sample_dilate_iterations", 1))
    sample_mask = erode_mask(sample_mask, sample_erode_iterations, cv2, np)
    sample_mask = dilate_mask(sample_mask, sample_dilate_iterations, cv2, np)

    reference_mask = cv2.imread(str(REFERENCE_MASK), cv2.IMREAD_GRAYSCALE)
    if reference_mask is None:
        raise FileNotFoundError(f"Reference mask not found: {REFERENCE_MASK}")

    if reference_mask.shape != sample_mask.shape:
        raise ValueError(
            f"Reference mask shape {reference_mask.shape} does not match sample mask shape {sample_mask.shape}."
        )

    aligned_sample_mask, best_angle_deg, best_shift_x, best_shift_y = align_sample_mask(
        sample_mask,
        reference_mask,
        config.get("alignment", {}),
        cv2,
        np,
    )
    reference_allowed, reference_required = build_reference_regions(
        reference_mask,
        inspection_cfg,
        lambda mask, iterations: dilate_mask(mask, iterations, cv2, np),
        lambda mask, iterations: erode_mask(mask, iterations, cv2, np),
    )
    section_masks = compute_section_masks(
        reference_required,
        int(inspection_cfg.get("section_columns", 12)),
        cv2,
        np,
    )
    metrics = score_sample(reference_allowed, reference_required, aligned_sample_mask, section_masks)
    passed, threshold_summary = evaluate_metrics(metrics, inspection_cfg)

    required_coverage = float(threshold_summary["required_coverage"])
    outside_allowed_ratio = float(threshold_summary["outside_allowed_ratio"])
    min_section_coverage = float(threshold_summary["min_section_coverage"])

    min_required_coverage = float(threshold_summary["min_required_coverage"])
    max_outside_allowed_ratio = float(threshold_summary["max_outside_allowed_ratio"])
    min_section_coverage_limit = float(threshold_summary["min_section_coverage_limit"])

    required_white = reference_required > 0
    allowed_white = reference_allowed > 0
    diff = np.zeros((aligned_sample_mask.shape[0], aligned_sample_mask.shape[1], 3), dtype=np.uint8)
    diff[allowed_white] = (0, 80, 0)
    diff[required_white] = (0, 255, 0)
    diff[metrics["missing_required_mask"]] = (0, 0, 255)
    diff[metrics["outside_allowed_mask"]] = (255, 0, 0)

    debug_paths = {}
    if bool(inspection_cfg.get("save_debug_images", True)):
        stem = image_path.stem
        debug_paths = save_debug_outputs(stem, aligned_sample_mask, diff)

    details = {
        "roi": {
            "x": roi[0],
            "y": roi[1],
            "width": roi[2],
            "height": roi[3],
        },
        "best_angle_deg": best_angle_deg,
        "best_shift_x": best_shift_x,
        "best_shift_y": best_shift_y,
        "required_coverage": required_coverage,
        "outside_allowed_ratio": outside_allowed_ratio,
        "min_section_coverage": min_section_coverage,
        "section_coverages": metrics["section_coverages"],
        "sample_white_pixels": metrics["sample_white_pixels"],
        "min_required_coverage": min_required_coverage,
        "max_outside_allowed_ratio": max_outside_allowed_ratio,
        "min_section_coverage_limit": min_section_coverage_limit,
        "debug_paths": debug_paths,
    }
    return passed, details


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
        passed, details = inspect_against_reference(config, image_path)
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

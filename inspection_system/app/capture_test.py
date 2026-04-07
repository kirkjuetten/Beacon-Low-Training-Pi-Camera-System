#!/usr/bin/env python3
import sys
import time
from pathlib import Path

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inspection_system.app.alignment_utils import align_sample_mask
from inspection_system.app.inspection_pipeline import inspect_against_reference
from inspection_system.app.morphology_utils import dilate_mask, erode_mask
from inspection_system.app.preprocessing_utils import get_roi, make_binary_mask
from inspection_system.app.reference_region_utils import build_reference_regions
from inspection_system.app.scoring_utils import evaluate_metrics, score_sample
from inspection_system.app.section_mask_utils import compute_section_masks
from inspection_system.app.frame_acquisition import cleanup_temp_image, capture_to_temp
from inspection_system.app.camera_interface import REFERENCE_MASK, REFERENCE_IMAGE, import_cv2_and_numpy, create_project, switch_project, get_current_project, list_projects, load_config, IndicatorLED


def run_interactive_training(config: dict) -> int:
    """Import and run interactive training mode."""
    try:
        from inspection_system.app.interactive_training import run_interactive_training as train_func
        return train_func(config)
    except ImportError as e:
        print(f"Interactive training not available: {e}")
        return 1


def save_debug_outputs(stem: str, aligned_sample_mask, diff_image) -> dict:
    cv2, _ = import_cv2_and_numpy()
    debug_mask_path = Path(REFERENCE_MASK).parent / f"{stem}_mask.png"
    debug_diff_path = Path(REFERENCE_MASK).parent / f"{stem}_diff.png"
    cv2.imwrite(str(debug_mask_path), aligned_sample_mask)
    cv2.imwrite(str(debug_diff_path), diff_image)
    return {
        "mask": str(debug_mask_path),
        "diff": str(debug_diff_path),
    }


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


def run_capture_and_inspect(config: dict, indicator) -> int:
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
        if mode == "train":
            return run_interactive_training(config)

        # Project management commands
        if mode == "create-project":
            if len(sys.argv) < 3:
                print("Usage: python3 capture_test.py create-project <project_name> [description]")
                return 2
            project_name = sys.argv[2]
            description = sys.argv[3] if len(sys.argv) > 3 else ""
            if create_project(project_name, description):
                print(f"Created project '{project_name}'")
                return 0
            else:
                print(f"Failed to create project '{project_name}'")
                return 1

        if mode == "switch-project":
            if len(sys.argv) < 3:
                print("Usage: python3 capture_test.py switch-project <project_name>")
                return 2
            project_name = sys.argv[2]
            if switch_project(project_name):
                print(f"Switched to project '{project_name}'")
                return 0
            else:
                print(f"Failed to switch to project '{project_name}'")
                return 1

        if mode == "list-projects":
            projects = list_projects()
            current = get_current_project()
            print(f"Current project: {current or 'None'}")
            print("Available projects:")
            for project in projects:
                status = " (ACTIVE)" if project["is_current"] else ""
                print(f"  {project['name']}{status}: {project['description']}")
            return 0

        if mode == "project-manager":
            # Launch GUI project manager
            try:
                from inspection_system.app.project_manager import main as pm_main
                pm_main()
                return 0
            except ImportError as e:
                print(f"GUI not available: {e}")
                print("Install tkinter and pygame for GUI support")
                return 1

        if mode == "dashboard":
            try:
                from inspection_system.app.operator_dashboard import main as dashboard_main
                dashboard_main()
                return 0
            except ImportError as e:
                print(f"Dashboard not available: {e}")
                print("Install tkinter for dashboard support")
                return 1

        print("Usage:")
        print("  python3 capture_test.py capture")
        print("  python3 capture_test.py set-reference")
        print("  python3 capture_test.py inspect")
        print("  python3 capture_test.py train")
        print("  python3 capture_test.py dashboard")
        print("  python3 capture_test.py create-project <name> [description]")
        print("  python3 capture_test.py switch-project <name>")
        print("  python3 capture_test.py list-projects")
        print("  python3 capture_test.py project-manager  # GUI")
        return 2
    finally:
        indicator.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

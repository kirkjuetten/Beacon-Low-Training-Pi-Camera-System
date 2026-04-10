#!/usr/bin/env python3
"""
Interactive Training Interface for Beacon Inspection System

Provides a GUI for real-time inspection with human-in-the-loop feedback
to train and adjust inspection thresholds.
"""

import json
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List

try:
    import pygame
    import pygame.gfxdraw
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from inspection_system.app.camera_interface import (
    load_config,
    import_cv2_and_numpy,
    IndicatorLED,
    get_active_runtime_paths,
)
from inspection_system.app.frame_acquisition import cleanup_temp_image, capture_to_temp
from inspection_system.app.inspection_pipeline import inspect_against_reference
from inspection_system.app.alignment_utils import align_sample_mask
from inspection_system.app.morphology_utils import dilate_mask, erode_mask
from inspection_system.app.preprocessing_utils import make_binary_mask
from inspection_system.app.reference_region_utils import build_reference_regions
from inspection_system.app.scoring_utils import evaluate_metrics, score_sample
from inspection_system.app.section_mask_utils import compute_section_masks
from inspection_system.app.capture_test import save_debug_outputs


class InspectionDisplay:
    """GUI display for inspection results with interactive feedback."""

    def cleanup(self) -> None:
        """Clean up display resources."""
        try:
            pygame.quit()
        except Exception:
            pass

    def __init__(self, width: int = 1024, height: int = 768):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for interactive display. Install with: pip install pygame")

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Beacon Inspection Training")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.clock = pygame.time.Clock()

        # Colors
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)

        # Button definitions
        self.buttons = {
            'approve': pygame.Rect(width - 200, height - 100, 80, 40),
            'reject': pygame.Rect(width - 100, height - 100, 80, 40),
            'review': pygame.Rect(width - 300, height - 100, 80, 40),
        }

    def draw_image_with_border(self, surface: pygame.Surface, border_color: tuple, border_width: int = 10):
        """Draw an image with a colored border."""
        # Draw border
        border_rect = surface.get_rect()
        pygame.draw.rect(self.screen, border_color, border_rect, border_width)

        # Draw image
        self.screen.blit(surface, (border_width, border_width))

    def draw_buttons(self):
        """Draw interactive buttons."""
        # Approve button (green)
        pygame.draw.rect(self.screen, self.GREEN, self.buttons['approve'])
        approve_text = self.small_font.render("APPROVE", True, self.BLACK)
        self.screen.blit(approve_text, (self.buttons['approve'].x + 5, self.buttons['approve'].y + 10))

        # Reject button (red)
        pygame.draw.rect(self.screen, self.RED, self.buttons['reject'])
        reject_text = self.small_font.render("REJECT", True, self.BLACK)
        self.screen.blit(reject_text, (self.buttons['reject'].x + 10, self.buttons['reject'].y + 10))

        # Review button (yellow)
        pygame.draw.rect(self.screen, self.YELLOW, self.buttons['review'])
        review_text = self.small_font.render("REVIEW", True, self.BLACK)
        self.screen.blit(review_text, (self.buttons['review'].x + 5, self.buttons['review'].y + 10))

    def draw_metrics(self, details: dict, y_offset: int = 50):
        """Draw inspection metrics on screen."""
        metrics = [
            f"Required Coverage: {details.get('required_coverage', 0):.4f} (min {details.get('min_required_coverage', 0):.4f})",
            f"Outside Allowed: {details.get('outside_allowed_ratio', 0):.4f} (max {details.get('max_outside_allowed_ratio', 0):.4f})",
            f"Min Section: {details.get('min_section_coverage', 0):.4f} (min {details.get('min_section_coverage_limit', 0):.4f})",
            f"Angle Correction: {details.get('best_angle_deg', 0):.2f}°",
            f"Shift: x={details.get('best_shift_x', 0)}, y={details.get('best_shift_y', 0)}",
        ]

        if 'ssim' in details:
            metrics.append(f"SSIM: {details['ssim']:.4f}")
        if 'anomaly_score' in details:
            metrics.append(f"Anomaly Score: {details['anomaly_score']:.4f}")

    def generate_inspection_description(self, passed: bool, details: dict) -> str:
        """Generate human-readable description of inspection results."""
        descriptions = []

        if passed:
            descriptions.append("✓ APPROVED: Sample meets all quality requirements")
            return ". ".join(descriptions)

        # Analyze failure reasons
        req_cov = details.get('required_coverage', 0)
        min_req = details.get('min_required_coverage', 0)
        outside_ratio = details.get('outside_allowed_ratio', 0)
        max_outside = details.get('max_outside_allowed_ratio', 0)
        min_section = details.get('min_section_coverage', 0)
        min_section_limit = details.get('min_section_coverage_limit', 0)

        if req_cov < min_req:
            deficit = min_req - req_cov
            descriptions.append(f"✗ REQUIRED COVERAGE: Missing {deficit:.1%} of required features")

        if outside_ratio > max_outside:
            excess = outside_ratio - max_outside
            descriptions.append(f"✗ OUTSIDE ALLOWED: {excess:.1%} excess material in restricted areas")

        if min_section < min_section_limit:
            deficit = min_section_limit - min_section
            descriptions.append(f"✗ SECTION COVERAGE: Weakest section missing {deficit:.1%} coverage")

        # Check anomaly metrics
        if 'ssim' in details and details['ssim'] < 0.8:
            descriptions.append(f"✗ STRUCTURAL SIMILARITY: Low SSIM ({details['ssim']:.3f}) indicates poor match to reference")

        if 'anomaly_score' in details and details['anomaly_score'] > 0.5:
            descriptions.append(f"✗ ANOMALY DETECTED: High anomaly score ({details['anomaly_score']:.3f}) suggests defects")

        # Alignment issues
        angle_correction = abs(details.get('best_angle_deg', 0))
        shift_distance = ((details.get('best_shift_x', 0) ** 2 + details.get('best_shift_y', 0) ** 2) ** 0.5)

        if angle_correction > 0.5:
            descriptions.append(f"⚠ ALIGNMENT: Significant rotation correction ({angle_correction:.2f}°) applied")

        if shift_distance > 2:
            descriptions.append(f"⚠ ALIGNMENT: Large positional shift ({shift_distance:.1f} pixels) detected")

        if not descriptions:
            descriptions.append("✗ UNKNOWN FAILURE: Sample failed but no specific issues identified")

        return ". ".join(descriptions)

    def draw_description(self, description: str, y_offset: int = 200):
        """Draw the inspection description on screen."""
        # Split description into lines that fit the screen
        screen_width = self.screen.get_size()[0] - 20
        words = description.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            if self.small_font.size(test_line)[0] < screen_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # Draw each line
        for i, line in enumerate(lines[:6]):  # Limit to 6 lines
            text = self.small_font.render(line, True, self.WHITE)
            self.screen.blit(text, (10, y_offset + i * 22))

    def display_inspection(self, image_path: Path, passed: bool, details: dict, logger: Optional["TrainingLogger"] = None) -> Optional[str]:
        """Display inspection result and wait for user input."""
        cv2, _ = import_cv2_and_numpy()

        # Load and prepare image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None

        # Convert BGR to RGB for pygame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create pygame surface
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

        # Scale to fit screen while maintaining aspect ratio
        screen_width, screen_height = self.screen.get_size()
        img_width, img_height = surface.get_size()
        scale = min((screen_width - 20) / img_width, (screen_height - 250) / img_height)  # More space for description
        if scale < 1:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            surface = pygame.transform.smoothscale(surface, (new_width, new_height))

        # Determine border color and generate description
        if passed:
            border_color = self.GREEN
            status_text = "PASS"
        else:
            # Check if it should be flagged for review (close to thresholds)
            req_cov = details.get('required_coverage', 0)
            min_req = details.get('min_required_coverage', 0)
            outside_ratio = details.get('outside_allowed_ratio', 0)
            max_outside = details.get('max_outside_allowed_ratio', 0)

            # Flag for review if within 10% of thresholds
            if (req_cov >= min_req * 0.9) and (outside_ratio <= max_outside * 1.1):
                border_color = self.YELLOW
                status_text = "NEEDS REVIEW"
            else:
                border_color = self.RED
                status_text = "FAIL"

        # Generate human-readable description
        description = self.generate_inspection_description(passed, details)

        # Clear screen
        self.screen.fill(self.BLACK)

        # Draw image with border
        self.draw_image_with_border(surface, border_color)

        # Draw status
        status_surface = self.font.render(f"Status: {status_text}", True, border_color)
        self.screen.blit(status_surface, (10, 10))

        # Draw metrics
        self.draw_metrics(details)

        # Draw description
        self.draw_description(description)

        # Draw buttons
        self.draw_buttons()

        pygame.display.flip()

        # Wait for user input
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.buttons['approve'].collidepoint(mouse_pos):
                        if logger:
                            logger.log_inspection(image_path, passed, details, 'approve', description)
                        return 'approve'
                    elif self.buttons['reject'].collidepoint(mouse_pos):
                        if logger:
                            logger.log_inspection(image_path, passed, details, 'reject', description)
                        return 'reject'
                    elif self.buttons['review'].collidepoint(mouse_pos):
                        if logger:
                            logger.log_inspection(image_path, passed, details, 'review', description)
                        return 'review'

            self.clock.tick(30)

        pygame.quit()


class TrainingLogger:
    """Logs training sessions and decisions for analysis."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.current_session = None
        self.session_start_time = None

    def start_session(self):
        """Start a new training session."""
        self.session_start_time = time.time()
        session_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.session_start_time))
        self.current_session = f"training_session_{session_id}.log"
        self._log(f"=== TRAINING SESSION STARTED: {session_id} ===")

    def log_inspection(self, image_path: Path, passed: bool, details: dict, feedback: str, description: str):
        """Log an inspection result and feedback."""
        if not self.current_session:
            self.start_session()

        timestamp = time.strftime("%H:%M:%S", time.localtime(time.time()))
        status = "PASS" if passed else "FAIL"

        log_entry = f"[{timestamp}] {status} -> {feedback.upper()}"
        log_entry += f" | {Path(image_path).name}"
        log_entry += f" | Coverage: {details.get('required_coverage', 0):.3f}"
        log_entry += f" | Outside: {details.get('outside_allowed_ratio', 0):.3f}"

        if 'ssim' in details:
            log_entry += f" | SSIM: {details['ssim']:.3f}"
        if 'anomaly_score' in details:
            log_entry += f" | Anomaly: {details['anomaly_score']:.3f}"

        log_entry += f" | {description}"

        self._log(log_entry)

    def log_threshold_suggestion(self, suggestions: dict):
        """Log threshold adjustment suggestions."""
        if suggestions:
            self._log("=== THRESHOLD SUGGESTIONS ===")
            for key, value in suggestions.items():
                self._log(f"Suggested {key}: {value:.4f}")
            self._log("=" * 30)

    def end_session(self):
        """End the current training session."""
        if self.current_session:
            duration = time.time() - self.session_start_time
            self._log(f"=== SESSION ENDED: Duration {duration:.1f}s ===")
            self.current_session = None

    def _log(self, message: str):
        """Write message to current session log."""
        if self.current_session:
            log_path = self.log_dir / self.current_session
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    def get_session_summary(self) -> Dict[str, int]:
        """Get summary of current session decisions."""
        if not self.current_session:
            return {}

        log_path = self.log_dir / self.current_session
        if not log_path.exists():
            return {}

        summary = {'approve': 0, 'reject': 0, 'review': 0, 'total': 0}

        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '-> APPROVE' in line:
                    summary['approve'] += 1
                elif '-> REJECT' in line:
                    summary['reject'] += 1
                elif '-> REVIEW' in line:
                    summary['review'] += 1
                summary['total'] += 1 if any(x in line for x in ['PASS', 'FAIL']) else 0

        return summary


class ThresholdTrainer:
    """Manages threshold adjustment based on human feedback."""

    def __init__(self, config_path: Path, logger: Optional[TrainingLogger] = None):
        self.config_path = config_path
        self.logger = logger
        self.training_data = []
        self.load_training_data()

    def load_training_data(self):
        """Load existing training data."""
        training_file = self.config_path.parent / "training_data.json"
        if training_file.exists():
            with open(training_file, 'r') as f:
                self.training_data = json.load(f)

    def save_training_data(self):
        """Save training data."""
        training_file = self.config_path.parent / "training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=2)

    def record_feedback(self, details: dict, feedback: str):
        """Record human feedback for threshold adjustment."""
        record = {
            'timestamp': time.time(),
            'feedback': feedback,
            'metrics': {
                'required_coverage': details.get('required_coverage', 0),
                'outside_allowed_ratio': details.get('outside_allowed_ratio', 0),
                'min_section_coverage': details.get('min_section_coverage', 0),
                'ssim': details.get('ssim'),
                'anomaly_score': details.get('anomaly_score'),
            }
        }
        self.training_data.append(record)
        self.save_training_data()

    def suggest_thresholds(self) -> dict:
        """Analyze training data and suggest new thresholds."""
        if len(self.training_data) < 10:
            return {}  # Need more data

        approved = [r for r in self.training_data if r['feedback'] == 'approve']
        rejected = [r for r in self.training_data if r['feedback'] == 'reject']

        if not approved or not rejected:
            return {}

        suggestions = {}

        # Adjust required coverage threshold
        approved_cov = [r['metrics']['required_coverage'] for r in approved]
        rejected_cov = [r['metrics']['required_coverage'] for r in rejected]

        if approved_cov and rejected_cov:
            min_approved = min(approved_cov)
            max_rejected = max(rejected_cov)
            if min_approved > max_rejected:
                suggestions['min_required_coverage'] = (min_approved + max_rejected) / 2

        # Adjust outside allowed ratio threshold
        approved_ratio = [r['metrics']['outside_allowed_ratio'] for r in approved]
        rejected_ratio = [r['metrics']['outside_allowed_ratio'] for r in rejected]

        if approved_ratio and rejected_ratio:
            max_approved = max(approved_ratio)
            min_rejected = min(rejected_ratio)
            if max_approved < min_rejected:
                suggestions['max_outside_allowed_ratio'] = (max_approved + min_rejected) / 2

        if self.logger and suggestions:
            self.logger.log_threshold_suggestion(suggestions)

        return suggestions

    def apply_suggestions(self, config: dict, suggestions: dict) -> dict:
        """Persist suggested threshold changes and update the in-memory config."""
        if not suggestions:
            return {}

        inspection_cfg = config.setdefault('inspection', {})

        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
        else:
            file_config = config.copy()

        file_inspection_cfg = file_config.setdefault('inspection', {})
        applied = {}

        for key, value in suggestions.items():
            normalized_value = round(float(value), 4)
            if float(inspection_cfg.get(key, normalized_value)) == normalized_value:
                continue
            inspection_cfg[key] = normalized_value
            file_inspection_cfg[key] = normalized_value
            applied[key] = normalized_value

        if applied:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(file_config, f, indent=2)
                f.write('\n')

        return applied


def run_interactive_training(config: dict) -> int:
    """Run interactive training mode."""
    if not PYGAME_AVAILABLE:
        print("pygame is required for interactive training. Install with: pip install pygame")
        return 1

    active_paths = get_active_runtime_paths()

    # Initialize logger
    log_dir = active_paths["log_dir"]
    logger = TrainingLogger(log_dir)
    logger.start_session()

    display = InspectionDisplay()
    trainer = ThresholdTrainer(active_paths["config_file"], logger)

    led_cfg = config.get("indicator_led", {})
    indicator = IndicatorLED(
        enabled=bool(led_cfg.get("enabled", False)),
        pass_gpio=int(led_cfg.get("pass_gpio", 23)),
        fail_gpio=int(led_cfg.get("fail_gpio", 24)),
        pulse_ms=int(led_cfg.get("pulse_ms", 750)),
    )

    try:
        print("Starting interactive training mode...")
        print("Controls:")
        print("- Green APPROVE button: Accept the sample")
        print("- Red REJECT button: Reject the sample")
        print("- Yellow REVIEW button: Flag for human review")
        print("- Close window to exit")
        print("\nDetailed descriptions will appear on screen for each inspection.")

        session_count = 0

        while True:
            # Capture image
            result_code, image_path, stderr_text = capture_to_temp(config)
            if result_code != 0:
                print(f"Capture failed: {stderr_text}")
                time.sleep(2)
                continue

            try:
                # Run inspection
                passed, details = inspect_against_reference(
                    config,
                    image_path,
                    make_binary_mask,
                    active_paths["reference_mask"],
                    active_paths["reference_image"],
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

                # Display result and get feedback
                feedback = display.display_inspection(image_path, passed, details, logger)

                if feedback == 'quit':
                    break
                elif feedback in ['approve', 'reject', 'review']:
                    trainer.record_feedback(details, feedback)
                    session_count += 1

                    # Update indicators
                    if feedback == 'approve':
                        indicator.pulse_pass()
                    elif feedback == 'reject':
                        indicator.pulse_fail()
                    # No indicator for review

                    # Show session summary every 10 samples
                    if session_count % 10 == 0:
                        summary = logger.get_session_summary()
                        print(f"\nSession Summary (last {summary.get('total', 0)} samples):")
                        print(f"  Approved: {summary.get('approve', 0)}")
                        print(f"  Rejected: {summary.get('reject', 0)}")
                        print(f"  Flagged for review: {summary.get('review', 0)}")

                    # Apply threshold suggestions periodically to avoid noisy rewrites.
                    if session_count % 10 == 0:
                        suggestions = trainer.suggest_thresholds()
                        applied = trainer.apply_suggestions(config, suggestions)
                        if applied:
                            print("\nApplied threshold updates based on training data:")
                            for key, value in applied.items():
                                print(f"  {key}: {value:.4f}")

            finally:
                cleanup_temp_image()

    finally:
        logger.end_session()
        display.cleanup()
        indicator.cleanup()

        # Final session summary
        summary = logger.get_session_summary()
        if summary.get('total', 0) > 0:
            print("\nFinal Session Summary:")
            print(f"  Total samples reviewed: {summary.get('total', 0)}")
            print(f"  Approved: {summary.get('approve', 0)}")
            print(f"  Rejected: {summary.get('reject', 0)}")
            print(f"  Flagged for review: {summary.get('review', 0)}")
            print(f"\nTraining logs saved to: {log_dir}")

    return 0


def main() -> int:
    config = load_config()
    return run_interactive_training(config)


if __name__ == "__main__":
    raise SystemExit(main())

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

    MIN_WIDTH = 640
    MIN_HEIGHT = 420

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
        display_info = pygame.display.Info()
        max_width = max(self.MIN_WIDTH, display_info.current_w - 40)
        max_height = max(self.MIN_HEIGHT, display_info.current_h - 80)
        initial_width = min(max(width, self.MIN_WIDTH), max_width)
        initial_height = min(max(height, self.MIN_HEIGHT), max_height)

        self.screen = pygame.display.set_mode((initial_width, initial_height), pygame.RESIZABLE)
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

        self.buttons: Dict[str, pygame.Rect] = {}
        self.layout: Dict[str, pygame.Rect] = {}
        self._reflow_layout()

    @staticmethod
    def _clamp(value: int, lower: int, upper: int) -> int:
        return max(lower, min(value, upper))

    def _update_fonts(self) -> None:
        _, height = self.screen.get_size()
        base_size = self._clamp(int(height * 0.035), 20, 34)
        small_size = self._clamp(int(height * 0.026), 15, 26)
        self.font = pygame.font.Font(None, base_size)
        self.small_font = pygame.font.Font(None, small_size)

    def _reflow_layout(self) -> None:
        width, height = self.screen.get_size()
        pad = self._clamp(int(height * 0.02), 8, 20)

        status_h = self._clamp(int(height * 0.07), 30, 56)
        controls_h = self._clamp(int(height * 0.17), 70, 140)
        metrics_h = self._clamp(int(height * 0.16), 70, 130)
        description_h = self._clamp(int(height * 0.18), 70, 140)

        available_for_image = height - (status_h + controls_h + metrics_h + description_h + pad * 5)
        image_h = max(80, available_for_image)

        status_rect = pygame.Rect(pad, pad, width - pad * 2, status_h)
        image_rect = pygame.Rect(pad, status_rect.bottom + pad, width - pad * 2, image_h)
        metrics_rect = pygame.Rect(pad, image_rect.bottom + pad, width - pad * 2, metrics_h)
        description_rect = pygame.Rect(pad, metrics_rect.bottom + pad, width - pad * 2, description_h)
        controls_rect = pygame.Rect(pad, height - controls_h - pad, width - pad * 2, controls_h)

        self.layout = {
            "status_rect": status_rect,
            "image_rect": image_rect,
            "metrics_rect": metrics_rect,
            "description_rect": description_rect,
            "controls_rect": controls_rect,
        }

        self._update_fonts()
        self._layout_buttons(controls_rect)

    def _layout_buttons(self, controls_rect: pygame.Rect) -> None:
        pad = self._clamp(int(controls_rect.height * 0.14), 6, 16)
        gap = self._clamp(int(controls_rect.width * 0.02), 8, 24)
        button_h = self._clamp(int(controls_rect.height * 0.58), 36, 56)
        # 4 buttons: REVIEW | APPROVE | REJECT | SET REF
        button_w = self._clamp(int(controls_rect.width * 0.20), 70, 180)
        needed_width = button_w * 4 + gap * 3

        if needed_width <= controls_rect.width - pad * 2:
            x_start = controls_rect.x + (controls_rect.width - needed_width) // 2
            y = controls_rect.y + (controls_rect.height - button_h) // 2
            self.buttons = {
                "review": pygame.Rect(x_start, y, button_w, button_h),
                "approve": pygame.Rect(x_start + button_w + gap, y, button_w, button_h),
                "reject": pygame.Rect(x_start + (button_w + gap) * 2, y, button_w, button_h),
                "set_ref": pygame.Rect(x_start + (button_w + gap) * 3, y, button_w, button_h),
            }
            return

        # Fall back to vertical layout for very narrow displays.
        stack_w = self._clamp(controls_rect.width - pad * 2, 80, 200)
        stack_gap = self._clamp(int(controls_rect.height * 0.06), 3, 10)
        stack_h = self._clamp((controls_rect.height - stack_gap * 3) // 4, 22, 40)
        x = controls_rect.x + (controls_rect.width - stack_w) // 2
        y0 = controls_rect.y + (controls_rect.height - (stack_h * 4 + stack_gap * 3)) // 2
        self.buttons = {
            "review": pygame.Rect(x, y0, stack_w, stack_h),
            "approve": pygame.Rect(x, y0 + stack_h + stack_gap, stack_w, stack_h),
            "reject": pygame.Rect(x, y0 + (stack_h + stack_gap) * 2, stack_w, stack_h),
            "set_ref": pygame.Rect(x, y0 + (stack_h + stack_gap) * 3, stack_w, stack_h),
        }

    def draw_image_with_border(self, surface: pygame.Surface, border_color: tuple, image_rect: pygame.Rect, border_width: int = 4):
        """Draw an image with a colored border."""
        draw_rect = surface.get_rect()
        draw_rect.center = image_rect.center
        draw_rect.clamp_ip(image_rect)
        border_rect = draw_rect.inflate(border_width * 2, border_width * 2)
        pygame.draw.rect(self.screen, border_color, border_rect, border_width)
        self.screen.blit(surface, draw_rect.topleft)

    def _scale_surface_to_rect(self, surface: pygame.Surface, target_rect: pygame.Rect) -> pygame.Surface:
        img_width, img_height = surface.get_size()
        available_w = max(1, target_rect.width - 8)
        available_h = max(1, target_rect.height - 8)
        scale = min(available_w / img_width, available_h / img_height)
        scale = min(scale, 1.0)
        new_width = max(1, int(img_width * scale))
        new_height = max(1, int(img_height * scale))
        return pygame.transform.smoothscale(surface, (new_width, new_height))

    def draw_buttons(self):
        """Draw interactive buttons."""
        BLUE = (70, 130, 220)
        button_colors = {
            "approve": self.GREEN,
            "reject": self.RED,
            "review": self.YELLOW,
            "set_ref": BLUE,
        }
        button_labels = {
            "approve": "APPROVE",
            "reject": "REJECT",
            "review": "REVIEW",
            "set_ref": "SET REF",
        }

        for key in ["review", "approve", "reject", "set_ref"]:
            button_rect = self.buttons[key]
            pygame.draw.rect(self.screen, button_colors[key], button_rect, border_radius=6)
            label_color = self.WHITE if key == "set_ref" else self.BLACK
            text = self.small_font.render(button_labels[key], True, label_color)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)

    def draw_metrics(self, details: dict, area: pygame.Rect):
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

        y = area.y
        line_height = self.small_font.get_linesize() + 2
        max_lines = max(1, area.height // line_height)
        for line in metrics[:max_lines]:
            text = self.small_font.render(line, True, self.WHITE)
            self.screen.blit(text, (area.x, y))
            y += line_height

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

    def draw_description(self, description: str, area: pygame.Rect):
        """Draw the inspection description on screen."""
        # Split description into lines that fit the screen
        screen_width = area.width
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
        line_height = self.small_font.get_linesize() + 2
        max_lines = max(1, area.height // line_height)
        for i, line in enumerate(lines[:max_lines]):
            text = self.small_font.render(line, True, self.WHITE)
            self.screen.blit(text, (area.x, area.y + i * line_height))

    def show_message(self, message: str, color: Optional[tuple] = None) -> None:
        """Fill screen with a centred status message (used for brief feedback)."""
        self._reflow_layout()
        self.screen.fill(self.BLACK)
        col = color or self.WHITE
        text = self.font.render(message, True, col)
        text_rect = text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

    def prompt_set_reference(self) -> str:
        """Show a full-screen prompt asking the operator to capture a reference.

        Returns 'capture' when the operator presses the capture button, or
        'quit' if they close the window or press Escape.
        """
        self._reflow_layout()
        width, height = self.screen.get_size()
        pad = self._clamp(int(height * 0.04), 12, 40)
        btn_w = self._clamp(int(width * 0.35), 160, 360)
        btn_h = self._clamp(int(height * 0.13), 44, 80)
        capture_btn = pygame.Rect(
            (width - btn_w) // 2,
            height // 2 + pad,
            btn_w,
            btn_h,
        )

        def render() -> None:
            self.screen.fill(self.BLACK)
            lines = [
                "No reference found for this project.",
                "Point the camera at the golden reference sample,",
                "then press CAPTURE REFERENCE.",
            ]
            line_h = self.font.get_linesize() + 4
            y_start = height // 2 - (len(lines) * line_h) - pad
            for i, line in enumerate(lines):
                surf = self.font.render(line, True, self.YELLOW)
                rect = surf.get_rect(centerx=width // 2, top=y_start + i * line_h)
                self.screen.blit(surf, rect)
            BLUE = (70, 130, 220)
            pygame.draw.rect(self.screen, BLUE, capture_btn, border_radius=8)
            label = self.font.render("CAPTURE REFERENCE", True, self.WHITE)
            self.screen.blit(label, label.get_rect(center=capture_btn.center))
            pygame.display.flip()

        render()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                elif event.type == pygame.VIDEORESIZE:
                    new_w = max(event.w, self.MIN_WIDTH)
                    new_h = max(event.h, self.MIN_HEIGHT)
                    self.screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)
                    width, height = self.screen.get_size()
                    pad = self._clamp(int(height * 0.04), 12, 40)
                    btn_w = self._clamp(int(width * 0.35), 160, 360)
                    btn_h = self._clamp(int(height * 0.13), 44, 80)
                    capture_btn = pygame.Rect((width - btn_w) // 2, height // 2 + pad, btn_w, btn_h)
                    render()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if capture_btn.collidepoint(event.pos):
                        return 'capture'
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return 'quit'
            self.clock.tick(30)

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
        source_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

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

        def render_frame() -> None:
            self._reflow_layout()
            self.screen.fill(self.BLACK)

            status_rect = self.layout["status_rect"]
            image_rect = self.layout["image_rect"]
            metrics_rect = self.layout["metrics_rect"]
            description_rect = self.layout["description_rect"]

            surface = self._scale_surface_to_rect(source_surface, image_rect)
            self.draw_image_with_border(surface, border_color, image_rect)

            status_surface = self.font.render(f"Status: {status_text}", True, border_color)
            self.screen.blit(status_surface, status_rect.topleft)

            self.draw_metrics(details, metrics_rect)
            self.draw_description(description, description_rect)
            self.draw_buttons()
            pygame.display.flip()

        render_frame()

        # Wait for user input
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                elif event.type == pygame.VIDEORESIZE:
                    new_width = max(event.w, self.MIN_WIDTH)
                    new_height = max(event.h, self.MIN_HEIGHT)
                    self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                    render_frame()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
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
                    elif self.buttons['set_ref'].collidepoint(mouse_pos):
                        return 'set_ref'
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return 'quit'

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


def capture_reference(config: dict) -> tuple[bool, str]:
    """Capture a new golden reference for the active project.

    Returns (success, message).
    """
    result_code, image_path, stderr_text = capture_to_temp(config)
    if result_code != 0:
        msg = f"Reference capture failed: {stderr_text}"
        cleanup_temp_image()
        return False, msg

    try:
        cv2, np = import_cv2_and_numpy()
        inspection_cfg = config.get("inspection", {})
        roi_image, _, mask, _, _, _ = make_binary_mask(image_path, inspection_cfg, import_cv2_and_numpy)
        ref_erode = int(inspection_cfg.get("reference_erode_iterations", 1))
        ref_dilate = int(inspection_cfg.get("reference_dilate_iterations", 1))
        mask = erode_mask(mask, ref_erode, cv2, np)
        mask = dilate_mask(mask, ref_dilate, cv2, np)

        white_pixels = int((mask > 0).sum())
        min_white = int(inspection_cfg.get("min_white_pixels", 100))
        if white_pixels < min_white:
            return False, f"Too few white pixels ({white_pixels}). Adjust ROI or threshold."

        active_paths = get_active_runtime_paths()
        ref_mask_path = active_paths["reference_mask"]
        ref_image_path = active_paths["reference_image"]
        ref_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ref_mask_path), mask)
        cv2.imwrite(str(ref_image_path), roi_image)
        return True, f"Reference saved ({white_pixels} white pixels)"
    except Exception as exc:
        return False, f"Reference capture error: {exc}"
    finally:
        cleanup_temp_image()


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
        print("- Blue SET REF button: Capture a new golden reference")
        print("- Close window or Esc to exit")
        print("\nDetailed descriptions will appear on screen for each inspection.")

        session_count = 0

        # If no reference exists yet, prompt operator to capture one first.
        if not active_paths["reference_mask"].exists():
            print("No reference mask found. Prompting operator to capture reference.")
            action = display.prompt_set_reference()
            if action == 'quit':
                return 0
            display.show_message("Capturing reference...", display.YELLOW)
            success, msg = capture_reference(config)
            print(msg)
            active_paths = get_active_runtime_paths()
            if not success:
                display.show_message(f"Failed: {msg}", display.RED)
                time.sleep(3)
                return 1
            display.show_message(f"Reference saved. Starting training...", display.GREEN)
            time.sleep(1)

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
                elif feedback == 'set_ref':
                    display.show_message("Capturing reference...", display.YELLOW)
                    success, msg = capture_reference(config)
                    print(msg)
                    active_paths = get_active_runtime_paths()
                    color = display.GREEN if success else display.RED
                    display.show_message(msg, color)
                    time.sleep(1.5)
                    continue
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

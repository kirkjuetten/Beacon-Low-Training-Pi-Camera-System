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
import select
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
from inspection_system.app.reference_service import save_debug_outputs, bake_reference_mask, save_reference_metadata
from inspection_system.app.anomaly_detection_utils import AnomalyDetector


class InspectionDisplay:
    """GUI display for inspection results with interactive feedback."""

    MIN_WIDTH = 640
    MIN_HEIGHT = 420
    ALIGNMENT_PROFILES = ["strict", "balanced", "forgiving"]

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
        self.reference_button_label = "SET REF"
        self.alignment_profile_label = "ALIGN BAL"
        self.active_mode = "setup_reference"
        self.visible_buttons: List[str] = []

        self.buttons: Dict[str, pygame.Rect] = {}
        self.layout: Dict[str, pygame.Rect] = {}
        self.set_ui_mode("setup_reference")
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

    def set_ui_mode(self, mode: str) -> None:
        """Set the visible controls for the current training state."""
        self.active_mode = mode
        if mode == "inspection":
            self.reference_button_label = "RESET REF"
            self.visible_buttons = ["review", "approve", "reject", "set_ref", "align_profile", "home"]
        else:
            self.reference_button_label = "SET REF"
            self.visible_buttons = ["set_ref", "home"]

    def set_alignment_profile_label(self, profile: str) -> None:
        profile_key = str(profile).strip().lower() or "balanced"
        if profile_key not in self.ALIGNMENT_PROFILES:
            profile_key = "balanced"
        self.alignment_profile_label = f"ALIGN {profile_key[:3].upper()}"

    def _reflow_layout(self) -> None:
        width, height = self.screen.get_size()
        pad = self._clamp(int(height * 0.02), 8, 20)

        status_h = self._clamp(int(height * 0.07), 30, 56)
        # Keep controls readable while prioritizing the camera image area.
        controls_ratio = 0.21 if self.active_mode == "inspection" else 0.15
        controls_h = self._clamp(int(height * controls_ratio), 90 if self.active_mode == "inspection" else 70, 180)
        metrics_h = self._clamp(int(height * 0.12), 58, 110)
        description_h = self._clamp(int(height * 0.13), 58, 115)

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
        visible_buttons = self.visible_buttons or ["set_ref"]

        if self.active_mode == "inspection" and len(visible_buttons) in {5, 6}:
            # Use two rows in inspection mode to keep touch targets large.
            gap_x = self._clamp(int(controls_rect.width * 0.02), 8, 24)
            gap_y = self._clamp(int(controls_rect.height * 0.10), 8, 18)

            row1_keys = ["review", "approve", "reject"]
            row2_keys = ["set_ref", "align_profile"]
            if "home" in visible_buttons:
                row2_keys.append("home")

            row1_h = self._clamp(int(controls_rect.height * 0.38), 34, 58)
            row2_h = self._clamp(int(controls_rect.height * 0.34), 32, 54)
            top_y = controls_rect.y + (controls_rect.height - (row1_h + gap_y + row2_h)) // 2
            row2_y = top_y + row1_h + gap_y

            row1_total_gap = gap_x * (len(row1_keys) - 1)
            row1_w = max(70, (controls_rect.width - row1_total_gap) // len(row1_keys))
            row1_total_w = row1_w * len(row1_keys) + row1_total_gap
            row1_x = controls_rect.x + (controls_rect.width - row1_total_w) // 2

            row2_total_gap = gap_x * (len(row2_keys) - 1)
            row2_w = max(90, (controls_rect.width - row2_total_gap) // len(row2_keys))
            row2_total_w = row2_w * len(row2_keys) + row2_total_gap
            row2_x = controls_rect.x + (controls_rect.width - row2_total_w) // 2

            self.buttons = {}
            x = row1_x
            for key in row1_keys:
                self.buttons[key] = pygame.Rect(x, top_y, row1_w, row1_h)
                x += row1_w + gap_x

            x = row2_x
            for key in row2_keys:
                self.buttons[key] = pygame.Rect(x, row2_y, row2_w, row2_h)
                x += row2_w + gap_x
            return

        pad = self._clamp(int(controls_rect.height * 0.14), 6, 16)
        gap = self._clamp(int(controls_rect.width * 0.02), 8, 24)
        button_h = self._clamp(int(controls_rect.height * 0.58), 36, 56)
        button_count = len(visible_buttons)
        button_fraction = 0.7 if button_count == 1 else 0.2
        button_w = self._clamp(int(controls_rect.width * button_fraction), 110 if button_count == 1 else 70, 320 if button_count == 1 else 180)
        needed_width = button_w * button_count + gap * max(0, button_count - 1)

        if needed_width <= controls_rect.width - pad * 2:
            x_start = controls_rect.x + (controls_rect.width - needed_width) // 2
            y = controls_rect.y + (controls_rect.height - button_h) // 2
            self.buttons = {}
            current_x = x_start
            for key in visible_buttons:
                self.buttons[key] = pygame.Rect(current_x, y, button_w, button_h)
                current_x += button_w + gap
            return

        # Fall back to vertical layout for very narrow displays.
        stack_w = self._clamp(controls_rect.width - pad * 2, 80, 200)
        stack_gap = self._clamp(int(controls_rect.height * 0.06), 3, 10)
        stack_h = self._clamp((controls_rect.height - stack_gap * max(0, button_count - 1)) // button_count, 22, 40)
        x = controls_rect.x + (controls_rect.width - stack_w) // 2
        total_stack_height = stack_h * button_count + stack_gap * max(0, button_count - 1)
        y0 = controls_rect.y + (controls_rect.height - total_stack_height) // 2
        self.buttons = {}
        current_y = y0
        for key in visible_buttons:
            self.buttons[key] = pygame.Rect(x, current_y, stack_w, stack_h)
            current_y += stack_h + stack_gap

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
        # Allow controlled upscaling for small ROI captures on larger touch displays.
        scale = min(scale, 2.4)
        new_width = max(1, int(img_width * scale))
        new_height = max(1, int(img_height * scale))
        return pygame.transform.smoothscale(surface, (new_width, new_height))

    def load_surface_from_image(self, image_path: Path) -> Optional[pygame.Surface]:
        """Load an image path into a pygame surface for display."""
        cv2, _ = import_cv2_and_numpy()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(image.swapaxes(0, 1))

    def draw_buttons(self):
        """Draw interactive buttons."""
        BLUE = (70, 130, 220)
        CYAN = (0, 170, 190)
        button_colors = {
            "approve": self.GREEN,
            "reject": self.RED,
            "review": self.YELLOW,
            "set_ref": BLUE,
            "align_profile": CYAN,
            "home": self.GRAY,
        }
        button_labels = {
            "approve": "APPROVE",
            "reject": "REJECT",
            "review": "REVIEW",
            "set_ref": self.reference_button_label,
            "align_profile": self.alignment_profile_label,
            "home": "HOME",
        }

        for key in self.visible_buttons:
            button_rect = self.buttons[key]
            pygame.draw.rect(self.screen, button_colors[key], button_rect, border_radius=6)
            label_color = self.WHITE if key in {"set_ref", "align_profile", "home"} else self.BLACK
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
        metrics.append(f"Alignment profile: {details.get('alignment_profile', 'balanced')}")

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

    def flash_action_confirmation(self, message: str, color: tuple, duration_ms: int = 450) -> bool:
        """Flash a short confirmation overlay. Returns True if user closes the window."""
        self._reflow_layout()
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        msg_surface = self.font.render(message, True, color)
        msg_rect = msg_surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        pad_x = 20
        pad_y = 14
        badge_rect = pygame.Rect(
            msg_rect.x - pad_x,
            msg_rect.y - pad_y,
            msg_rect.width + pad_x * 2,
            msg_rect.height + pad_y * 2,
        )
        pygame.draw.rect(self.screen, self.BLACK, badge_rect, border_radius=10)
        pygame.draw.rect(self.screen, color, badge_rect, 2, border_radius=10)
        self.screen.blit(msg_surface, msg_rect)
        pygame.display.flip()

        end_tick = pygame.time.get_ticks() + duration_ms
        while pygame.time.get_ticks() < end_tick:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return True
            self.clock.tick(60)
        return False

    def prompt_set_reference(self) -> str:
        """Deprecated compatibility wrapper for older callers."""
        return 'capture'

    def run_reference_preview(self, config: dict, has_reference: bool) -> str:
        """Show a live-ish preview loop until the operator captures/replaces the reference."""
        self.set_ui_mode("setup_reference")
        self.reference_button_label = "SET REF"
        last_surface: Optional[pygame.Surface] = None
        last_image_path: Optional[Path] = None
        status_color = self.YELLOW
        next_capture_time = 0.0

        def render() -> None:
            self._reflow_layout()
            self.screen.fill(self.BLACK)

            status_rect = self.layout["status_rect"]
            image_rect = self.layout["image_rect"]
            metrics_rect = self.layout["metrics_rect"]
            description_rect = self.layout["description_rect"]

            title = "Reference Preview"
            title_surface = self.font.render(title, True, status_color)
            self.screen.blit(title_surface, status_rect.topleft)

            if last_surface is not None:
                scaled_surface = self._scale_surface_to_rect(last_surface, image_rect)
                self.draw_image_with_border(scaled_surface, status_color, image_rect)
            else:
                placeholder = self.font.render("Waiting for camera preview...", True, self.WHITE)
                placeholder_rect = placeholder.get_rect(center=image_rect.center)
                self.screen.blit(placeholder, placeholder_rect)

            metric_lines = [
                f"Reference file: {'present' if has_reference else 'missing'}",
                f"Action: {self.reference_button_label}",
                "Camera preview refreshes automatically.",
            ]
            y = metrics_rect.y
            line_height = self.small_font.get_linesize() + 2
            for line in metric_lines:
                text = self.small_font.render(line, True, self.WHITE)
                self.screen.blit(text, (metrics_rect.x, y))
                y += line_height

            description = (
                "Point the camera at the golden reference sample. "
                f"Press {self.reference_button_label} when the framing looks correct."
            )
            self.draw_description(description, description_rect)
            self.draw_buttons()

            pygame.display.flip()

        while True:
            now = time.time()
            if now >= next_capture_time:
                result_code, image_path, stderr_text = capture_to_temp(config)
                if result_code == 0:
                    surface = self.load_surface_from_image(image_path)
                    if surface is not None:
                        last_surface = surface
                        last_image_path = image_path
                        status_color = self.GREEN if has_reference else self.YELLOW
                    else:
                        status_color = self.RED
                else:
                    status_color = self.RED
                next_capture_time = now + 0.35
                render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cleanup_temp_image()
                    return 'quit'
                elif event.type == pygame.VIDEORESIZE:
                    new_w = max(event.w, self.MIN_WIDTH)
                    new_h = max(event.h, self.MIN_HEIGHT)
                    self.screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)
                    render()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.buttons.get('set_ref') and self.buttons['set_ref'].collidepoint(event.pos):
                        return str(last_image_path) if last_image_path is not None else 'capture'
                    if self.buttons.get('home') and self.buttons['home'].collidepoint(event.pos):
                        return 'home'
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    cleanup_temp_image()
                    return 'quit'

            # Keep UI responsive before first camera frame arrives.
            if last_surface is None:
                render()

            self.clock.tick(30)

    def display_inspection(self, image_path: Path, passed: bool, details: dict, logger: Optional["TrainingLogger"] = None) -> Optional[str]:
        """Display inspection result and wait for user input."""
        source_surface = self.load_surface_from_image(image_path)
        if source_surface is None:
            return None
        self.set_ui_mode("inspection")
        self.set_alignment_profile_label(str(details.get("alignment_profile", "balanced")))

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
                        if self.flash_action_confirmation("APPROVED", self.GREEN):
                            return 'quit'
                        return 'approve'
                    elif self.buttons['reject'].collidepoint(mouse_pos):
                        if logger:
                            logger.log_inspection(image_path, passed, details, 'reject', description)
                        if self.flash_action_confirmation("REJECTED", self.RED):
                            return 'quit'
                        return 'reject'
                    elif self.buttons['review'].collidepoint(mouse_pos):
                        if logger:
                            logger.log_inspection(image_path, passed, details, 'review', description)
                        if self.flash_action_confirmation("MARKED FOR REVIEW", self.YELLOW):
                            return 'quit'
                        return 'review'
                    elif self.buttons['set_ref'].collidepoint(mouse_pos):
                        if self.flash_action_confirmation(self.reference_button_label, (70, 130, 220), duration_ms=300):
                            return 'quit'
                        return 'set_ref'
                    elif self.buttons['align_profile'].collidepoint(mouse_pos):
                        if self.flash_action_confirmation("ALIGNMENT PROFILE", (0, 170, 190), duration_ms=300):
                            return 'quit'
                        return 'align_profile'
                    elif self.buttons['home'].collidepoint(mouse_pos):
                        if self.flash_action_confirmation("RETURNING HOME", self.GRAY, duration_ms=300):
                            return 'quit'
                        return 'home'
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


def save_reference_from_image(config: dict, image_path: Path) -> tuple[bool, str]:
    """Create the active project's reference assets from an existing captured image."""
    roi_image, mask, feature_pixels, error_msg = bake_reference_mask(image_path, config)
    if error_msg:
        return False, error_msg

    try:
        cv2, _ = import_cv2_and_numpy()
        active_paths = get_active_runtime_paths()
        ref_mask_path = active_paths["reference_mask"]
        ref_image_path = active_paths["reference_image"]
        ref_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ref_mask_path), mask)
        cv2.imwrite(str(ref_image_path), roi_image)
        save_reference_metadata(config)
        return True, f"Reference saved ({feature_pixels} feature pixels)"
    except Exception as exc:
        return False, f"Reference save error: {exc}"


def cycle_alignment_profile(config: dict, config_path: Path) -> tuple[str, bool, str]:
    """Cycle alignment profile and persist it to the active project config."""
    profiles = ["strict", "balanced", "forgiving"]
    alignment_cfg = config.setdefault("alignment", {})
    current = str(alignment_cfg.get("tolerance_profile", "balanced")).strip().lower()
    if current not in profiles:
        current = "balanced"

    next_profile = profiles[(profiles.index(current) + 1) % len(profiles)]
    limits = {
        "strict": {"max_angle_deg": 0.7, "max_shift_x": 2, "max_shift_y": 2},
        "balanced": {"max_angle_deg": 1.0, "max_shift_x": 4, "max_shift_y": 3},
        "forgiving": {"max_angle_deg": 1.8, "max_shift_x": 7, "max_shift_y": 5},
    }[next_profile]

    alignment_cfg["tolerance_profile"] = next_profile
    alignment_cfg["max_angle_deg"] = limits["max_angle_deg"]
    alignment_cfg["max_shift_x"] = limits["max_shift_x"]
    alignment_cfg["max_shift_y"] = limits["max_shift_y"]

    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
        else:
            file_config = config.copy()

        file_alignment = file_config.setdefault("alignment", {})
        file_alignment["tolerance_profile"] = next_profile
        file_alignment["max_angle_deg"] = limits["max_angle_deg"]
        file_alignment["max_shift_x"] = limits["max_shift_x"]
        file_alignment["max_shift_y"] = limits["max_shift_y"]

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(file_config, f, indent=2)
            f.write("\n")

        return next_profile, True, f"Alignment profile set to {next_profile}"
    except Exception as exc:
        return current, False, f"Failed to save alignment profile: {exc}"


def run_interactive_training(config: dict) -> int:
    """Run interactive training mode."""
    if not PYGAME_AVAILABLE:
        print("pygame is required for interactive training. Install with: pip install pygame")
        return 1

    active_paths = get_active_runtime_paths()

    model_path = active_paths["reference_dir"] / "anomaly_model.pkl"
    anomaly_detector = None
    if model_path.exists():
        try:
            anomaly_detector = AnomalyDetector(model_path=model_path)
            anomaly_detector.load_model()
            print(f"Loaded anomaly model: {model_path}")
        except Exception as exc:
            print(f"Warning: failed to load anomaly model from {model_path}: {exc}")

    # Initialize logger
    log_dir = active_paths["log_dir"]
    logger = TrainingLogger(log_dir)
    logger.start_session()

    display = InspectionDisplay()
    trainer = ThresholdTrainer(active_paths["config_file"], logger)
    display.set_alignment_profile_label(config.get("alignment", {}).get("tolerance_profile", "balanced"))

    led_cfg = config.get("indicator_led", {})
    indicator = IndicatorLED(
        enabled=bool(led_cfg.get("enabled", False)),
        pass_gpio=int(led_cfg.get("pass_gpio", 23)),
        fail_gpio=int(led_cfg.get("fail_gpio", 24)),
        pulse_ms=int(led_cfg.get("pulse_ms", 750)),
    )

    training_cfg = config.get("training", {})
    early_review_parts = int(training_cfg.get("early_review_parts", 25))
    early_review_interval = int(training_cfg.get("early_review_interval", 5))
    steady_review_interval = int(training_cfg.get("steady_review_interval", 10))
    update_prompt_timeout_s = float(training_cfg.get("update_prompt_timeout_s", 5.0))

    def should_review_training(count: int) -> bool:
        if count <= 0:
            return False
        if count <= early_review_parts:
            return count % early_review_interval == 0
        return count % steady_review_interval == 0

    def prompt_apply_updates_with_timeout(timeout_s: float) -> bool:
        """Ask once for update approval without blocking indefinitely."""
        prompt = f"Apply suggested threshold updates now? [y/N] (auto-skip in {int(timeout_s)}s): "
        print(prompt, end="", flush=True)
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
            if ready:
                response = sys.stdin.readline().strip().lower()
                return response in {"y", "yes"}
        except Exception:
            # If stdin/select is unavailable, fail safe and skip auto-apply.
            pass
        print("(auto-skip)")
        return False

    try:
        print("Starting interactive training mode...")
        print("Controls:")
        print("- Green APPROVE button: Accept the sample")
        print("- Red REJECT button: Reject the sample")
        print("- Yellow REVIEW button: Flag for human review")
        print("- Blue SET REF button: Capture a new golden reference")
        print("- Gray HOME button: Return to dashboard/home screen")
        print("- Close window or Esc to exit")
        print("\nDetailed descriptions will appear on screen for each inspection.")

        session_count = 0

        # If no reference exists yet, show live preview until the operator captures one.
        if not active_paths["reference_mask"].exists():
            print("No reference mask found. Prompting operator to capture reference.")
            while True:
                action = display.run_reference_preview(config, has_reference=False)
                if action == 'home':
                    print("Returning to dashboard/home.")
                    return 0
                if action == 'quit':
                    return 0
                if action in {'capture', '', None}:
                    display.show_message("Waiting for preview frame...", display.YELLOW)
                    time.sleep(1)
                    continue
                display.show_message("Saving reference...", display.YELLOW)
                success, msg = save_reference_from_image(config, Path(action))
                cleanup_temp_image()
                print(msg)
                active_paths = get_active_runtime_paths()
                if success:
                    display.show_message("Reference saved. Starting training...", display.GREEN)
                    time.sleep(1)
                    break
                display.show_message(msg, display.RED)
                time.sleep(2)

        while True:
            # Capture image
            result_code, image_path, stderr_text = capture_to_temp(config)
            if result_code != 0:
                print(f"Capture failed: {stderr_text}")
                time.sleep(2)
                continue

            try:
                # Run inspection
                try:
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
                        anomaly_detector=anomaly_detector,
                    )
                except ValueError as exc:
                    err = str(exc)
                    print(f"Inspection error: {err}")
                    if "Reference mask shape" in err and "sample mask shape" in err:
                        display.show_message("Reference/sample size mismatch. Re-capture reference.", display.YELLOW)
                        time.sleep(1.2)
                        while True:
                            action = display.run_reference_preview(config, has_reference=active_paths["reference_mask"].exists())
                            if action == 'home':
                                print("Returning to dashboard/home.")
                                return 0
                            if action == 'quit':
                                return 0
                            if action in {'capture', '', None}:
                                display.show_message("Waiting for preview frame...", display.YELLOW)
                                time.sleep(1)
                                continue
                            display.show_message("Saving reference...", display.YELLOW)
                            success, msg = save_reference_from_image(config, Path(action))
                            cleanup_temp_image()
                            print(msg)
                            active_paths = get_active_runtime_paths()
                            color = display.GREEN if success else display.RED
                            display.show_message(msg, color)
                            time.sleep(1.5)
                            if success:
                                break
                        continue
                    display.show_message(f"Inspection error: {err}", display.RED)
                    time.sleep(1.2)
                    continue

                # Display result and get feedback
                feedback = display.display_inspection(image_path, passed, details, logger)

                if feedback == 'home':
                    print("Returning to dashboard/home.")
                    break
                if feedback == 'quit':
                    break
                elif feedback == 'set_ref':
                    go_home = False
                    while True:
                        action = display.run_reference_preview(config, has_reference=active_paths["reference_mask"].exists())
                        if action == 'home':
                            go_home = True
                            break
                        if action == 'quit':
                            break
                        if action in {'capture', '', None}:
                            display.show_message("Waiting for preview frame...", display.YELLOW)
                            time.sleep(1)
                            continue
                        display.show_message("Saving reference...", display.YELLOW)
                        success, msg = save_reference_from_image(config, Path(action))
                        cleanup_temp_image()
                        print(msg)
                        active_paths = get_active_runtime_paths()
                        color = display.GREEN if success else display.RED
                        display.show_message(msg, color)
                        time.sleep(1.5)
                        if success:
                            break
                    if go_home:
                        print("Returning to dashboard/home.")
                        break
                    continue
                elif feedback == 'align_profile':
                    profile, changed, msg = cycle_alignment_profile(config, active_paths["config_file"])
                    display.set_alignment_profile_label(profile)
                    print(msg)
                    color = display.GREEN if changed else display.RED
                    display.show_message(msg, color)
                    time.sleep(1.0)
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

                    # Show session summary at configured review cadence.
                    if should_review_training(session_count):
                        summary = logger.get_session_summary()
                        print(f"\nSession Summary (last {summary.get('total', 0)} samples):")
                        print(f"  Approved: {summary.get('approve', 0)}")
                        print(f"  Rejected: {summary.get('reject', 0)}")
                        print(f"  Flagged for review: {summary.get('review', 0)}")

                    # Review and optionally apply threshold suggestions at configured cadence.
                    if should_review_training(session_count):
                        suggestions = trainer.suggest_thresholds()
                        if suggestions:
                            print("\nSuggested threshold updates:")
                            for key, value in suggestions.items():
                                print(f"  {key}: {value:.4f}")

                            if prompt_apply_updates_with_timeout(update_prompt_timeout_s):
                                applied = trainer.apply_suggestions(config, suggestions)
                                if applied:
                                    print("Applied threshold updates:")
                                    for key, value in applied.items():
                                        print(f"  {key}: {value:.4f}")
                                else:
                                    print("No threshold changes were applied.")
                            else:
                                print("Skipped applying threshold updates for now.")

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

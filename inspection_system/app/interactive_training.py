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
from inspection_system.app.inspection_pipeline import inspect_against_references
from inspection_system.app.alignment_utils import align_sample_mask
from inspection_system.app.morphology_utils import dilate_mask, erode_mask
from inspection_system.app.preprocessing_utils import make_binary_mask
from inspection_system.app.reference_region_utils import build_reference_regions
from inspection_system.app.scoring_utils import evaluate_metrics, normalize_inspection_mode, score_sample
from inspection_system.app.section_mask_utils import compute_section_masks
from inspection_system.app.reference_service import (
    activate_reference_candidate,
    clear_reference_variants,
    discard_reference_candidate,
    list_runtime_reference_candidates,
    save_debug_outputs,
    bake_reference_mask,
    save_reference_metadata,
    stage_reference_candidate_from_image,
    check_reference_settings_match,
)
from inspection_system.app.runtime_controller import get_inspection_runtime_warnings, load_anomaly_detector


TRAINING_DATA_SCHEMA_VERSION = 2
LEARNED_RANGE_DIRECTIONS = {
    'required_coverage': 'higher_is_better',
    'outside_allowed_ratio': 'lower_is_better',
    'min_section_coverage': 'higher_is_better',
    'ssim': 'higher_is_better',
    'mse': 'lower_is_better',
    'anomaly_score': 'higher_is_better',
}
INSPECTION_MODE_OPTIONAL_METRICS = {
    'mask_only': set(),
    'mask_and_ssim': {'ssim', 'mse'},
    'mask_and_ml': {'anomaly_score'},
    'full': {'ssim', 'mse', 'anomaly_score'},
}


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
            self.visible_buttons = ["approve", "reject", "review", "capture", "set_ref"]
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

        if self.active_mode == "inspection":
            # New inspection layout: left sidebar + main image area + bottom text
            sidebar_w = self._clamp(int(width * 0.20), 140, 280)
            image_area_w = width - sidebar_w - pad * 3
            
            status_h = self._clamp(int(height * 0.06), 28, 50)
            description_h = self._clamp(int(height * 0.15), 70, 130)
            
            available_for_image = height - (status_h + description_h + pad * 4)
            image_h = max(80, available_for_image)
            
            status_rect = pygame.Rect(pad, pad, width - pad * 2, status_h)
            image_rect = pygame.Rect(sidebar_w + pad * 2, status_rect.bottom + pad, image_area_w, image_h)
            description_rect = pygame.Rect(pad, image_rect.bottom + pad, width - pad * 2, description_h)
            sidebar_rect = pygame.Rect(pad, status_rect.bottom + pad, sidebar_w, image_h)
            
            self.layout = {
                "status_rect": status_rect,
                "image_rect": image_rect,
                "metrics_rect": image_rect,  # Reuse image rect for metrics display overlay
                "description_rect": description_rect,
                "controls_rect": sidebar_rect,
                "sidebar_rect": sidebar_rect,
            }
            
            self._update_fonts()
            self._layout_buttons_sidebar(sidebar_rect)
            return
        
        # Original layout for setup/reference modes
        status_h = self._clamp(int(height * 0.07), 30, 56)
        controls_ratio = 0.15
        controls_h = self._clamp(int(height * controls_ratio), 70, 180)
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

    def _layout_buttons_sidebar(self, sidebar_rect: pygame.Rect) -> None:
        """Layout buttons in a sidebar: decision buttons first (large), utilities below."""
        decision_buttons = ["approve", "reject", "review"]
        utility_buttons = ["capture", "set_ref"]
        
        gap = self._clamp(int(sidebar_rect.height * 0.03), 6, 12)
        
        # Decision buttons: 3 buttons, take ~60% of sidebar height
        decision_h = self._clamp(int(sidebar_rect.height * 0.18), 50, 90)
        decisions_total_h = decision_h * 3 + gap * 2
        decisions_top = sidebar_rect.y + self._clamp(int(sidebar_rect.height * 0.04), 5, 15)
        
        # Utility buttons below: compress into remaining space
        utilities_available_h = sidebar_rect.height - decisions_total_h - gap * 2
        utility_h = self._clamp(int(utilities_available_h / 2) - gap, 36, 70)
        utilities_top = decisions_top + decisions_total_h + gap
        
        decision_w = sidebar_rect.width - self._clamp(int(sidebar_rect.width * 0.08), 4, 12)
        utility_w = decision_w
        
        self.buttons = {}
        
        # Decision buttons (larger, easy to tap)
        x = sidebar_rect.x + (sidebar_rect.width - decision_w) // 2
        for i, key in enumerate(decision_buttons):
            y = decisions_top + i * (decision_h + gap)
            self.buttons[key] = pygame.Rect(x, y, decision_w, decision_h)
        
        # Utility buttons (smaller)
        x = sidebar_rect.x + (sidebar_rect.width - utility_w) // 2
        for i, key in enumerate(utility_buttons):
            y = utilities_top + i * (utility_h + gap)
            self.buttons[key] = pygame.Rect(x, y, utility_w, utility_h)

    def _layout_buttons(self, controls_rect: pygame.Rect) -> None:
        visible_buttons = self.visible_buttons or ["set_ref"]

        if self.active_mode == "inspection" and len(visible_buttons) in {5, 6}:
            # Use two rows in inspection mode to keep touch targets large.
            gap_x = self._clamp(int(controls_rect.width * 0.02), 8, 24)
            gap_y = self._clamp(int(controls_rect.height * 0.10), 8, 18)

            row1_keys = ["review", "approve", "reject"]
            row2_keys = ["set_ref", "home"]

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

    def _make_processed_surface(
        self,
        image_path: Path,
        config: dict,
    ) -> Optional[pygame.Surface]:
        """Build a pygame surface showing the binary mask blended over the ROI image.
        Returns None on any error so callers can fall back to raw display."""
        try:
            inspection_cfg = (config or {}).get("inspection", {})
            roi_image, _gray, mask, _roi, cv2, np = make_binary_mask(
                image_path, inspection_cfg, import_cv2_and_numpy
            )
            roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
            overlay = roi_rgb.copy()
            green_tint = np.array([0, 200, 0], dtype=np.float32)
            in_mask = mask == 255
            overlay[in_mask] = np.clip(
                overlay[in_mask].astype(np.float32) * 0.5 + green_tint * 0.5,
                0, 255,
            ).astype(np.uint8)
            return pygame.surfarray.make_surface(overlay.swapaxes(0, 1))
        except Exception:
            return None

    def draw_buttons(self):
        """Draw interactive buttons. In inspection mode, add capture button and status."""
        BLUE = (70, 130, 220)
        CYAN = (0, 170, 190)
        
        button_colors = {
            "approve": self.GREEN,
            "reject": self.RED,
            "review": self.YELLOW,
            "capture": BLUE,
            "set_ref": CYAN,
            "home": self.GRAY,
        }

        button_labels = {
            "approve": "APPROVE",
            "reject": "REJECT",
            "review": "REVIEW",
            "capture": "CAPTURE",
            "set_ref": self.reference_button_label,
            "home": "HOME",
        }

        for key, rect in self.buttons.items():
            color = button_colors.get(key, self.GRAY)
            label = button_labels.get(key, key.upper())
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            
            # Draw label, respecting font size
            font = self.font if rect.height > 60 else self.small_font
            text = font.render(label, True, self.WHITE)
            text_rect = text.get_rect(center=rect.center)
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

    def display_inspection(
        self,
        image_path: Path,
        passed: bool,
        details: dict,
        logger: Optional["TrainingLogger"] = None,
        config: Optional[dict] = None,
        reference_mask_path: Optional[Path] = None,
    ) -> tuple[Optional[str], bool]:
        """Display inspection result and wait for user input.
        
        Returns: (feedback_action, ready_to_capture)
        feedback_action: 'approve', 'reject', 'review', 'set_ref', or 'quit'
        ready_to_capture: True once operator clicks CAPTURE after deciding feedback
        """
        source_surface = self.load_surface_from_image(image_path)
        if source_surface is None:
            return None, False
        self.set_ui_mode("inspection")
        self.set_alignment_profile_label(str(details.get("alignment_profile", "balanced")))

        # Read display mode and build processed surface if needed
        display_mode = (config or {}).get("inspection", {}).get("image_display_mode", "raw")
        processed_surface: Optional[pygame.Surface] = None
        if display_mode in ("processed", "split"):
            processed_surface = self._make_processed_surface(image_path, config or {})
            if processed_surface is None:
                display_mode = "raw"  # Fall back silently if processing fails

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

        # State: waiting for decision or waiting for capture trigger
        user_feedback = None

        def render_frame() -> None:
            self._reflow_layout()
            self.screen.fill(self.BLACK)

            status_rect = self.layout["status_rect"]
            image_rect = self.layout["image_rect"]
            description_rect = self.layout["description_rect"]

            if display_mode == "processed" and processed_surface is not None:
                display_surf = self._scale_surface_to_rect(processed_surface, image_rect)
            elif display_mode == "split" and processed_surface is not None:
                half_w = max(1, (image_rect.width - 4) // 2)
                split_h = max(1, image_rect.height - 8)
                try:
                    left = pygame.transform.smoothscale(source_surface, (half_w, split_h))
                    right = pygame.transform.smoothscale(processed_surface, (half_w, split_h))
                    combined = pygame.Surface((half_w * 2 + 4, split_h))
                    combined.fill((40, 40, 40))
                    combined.blit(left, (0, 0))
                    combined.blit(right, (half_w + 4, 0))
                    display_surf = combined
                except Exception:
                    display_surf = self._scale_surface_to_rect(source_surface, image_rect)
            else:
                display_surf = self._scale_surface_to_rect(source_surface, image_rect)
            self.draw_image_with_border(display_surf, border_color, image_rect)

            # Status line: show decision if made, ready state if awaiting capture
            if user_feedback:
                status_line = f"Status: {status_text} → Decision: {user_feedback.upper()} — Ready to capture next part"
                status_color = self.YELLOW
            else:
                status_line = f"Status: {status_text} — Click a decision button to proceed"
                status_color = border_color
            
            status_surface = self.font.render(status_line, True, status_color)
            self.screen.blit(status_surface, status_rect.topleft)

            # Draw metrics and description in right area
            self.draw_metrics(details, image_rect)  
            self.draw_description(description, description_rect)
            self.draw_buttons()
            pygame.display.flip()

        render_frame()

        # Two-phase loop: Phase 1 = get user decision; Phase 2 = wait for capture trigger
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit', False
                elif event.type == pygame.VIDEORESIZE:
                    new_width = max(event.w, self.MIN_WIDTH)
                    new_height = max(event.h, self.MIN_HEIGHT)
                    self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                    render_frame()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    
                    # Phase 1: Waiting for decision
                    if not user_feedback:
                        if self.buttons['approve'].collidepoint(mouse_pos):
                            if logger:
                                logger.log_inspection(image_path, passed, details, 'approve', description)
                            if self.flash_action_confirmation("APPROVED", self.GREEN):
                                return 'quit', False
                            user_feedback = 'approve'
                            render_frame()
                        elif self.buttons['reject'].collidepoint(mouse_pos):
                            if logger:
                                logger.log_inspection(image_path, passed, details, 'reject', description)
                            if self.flash_action_confirmation("REJECTED", self.RED):
                                return 'quit', False
                            user_feedback = 'reject'
                            render_frame()
                        elif self.buttons['review'].collidepoint(mouse_pos):
                            if logger:
                                logger.log_inspection(image_path, passed, details, 'review', description)
                            if self.flash_action_confirmation("MARKED FOR REVIEW", self.YELLOW):
                                return 'quit', False
                            user_feedback = 'review'
                            render_frame()
                        elif self.buttons['set_ref'].collidepoint(mouse_pos):
                            if self.flash_action_confirmation(self.reference_button_label, (70, 130, 220), duration_ms=300):
                                return 'quit', False
                            return 'set_ref', False
                    
                    # Phase 2: Decision made, waiting for CAPTURE button
                    else:
                        if self.buttons['capture'].collidepoint(mouse_pos):
                            if self.flash_action_confirmation("CAPTURING...", (70, 130, 220)):
                                return 'quit', False
                            return user_feedback, True  # Return decision + ready to capture
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return 'quit', False
                    elif event.key == pygame.K_RETURN and user_feedback:
                        # Allow pressing Enter to trigger capture after decision
                        return user_feedback, True

            self.clock.tick(30)

        pygame.quit()

    def show_training_checkpoint(
        self,
        summary: dict,
        suggestions: dict,
        review_warnings: Optional[List[str]] = None,
        learned_range_lines: Optional[List[str]] = None,
    ) -> str:
        """Pause training and ask operator how to handle pending learning data."""
        defer_color = (70, 130, 220)
        discard_color = self.RED
        update_color = self.GREEN
        home_color = self.GRAY

        button_rects: dict[str, pygame.Rect] = {}

        def render() -> None:
            self._reflow_layout()
            self.screen.fill(self.BLACK)
            width, height = self.screen.get_size()
            pad = self._clamp(int(height * 0.02), 10, 20)

            title = self.font.render("Training Review Checkpoint", True, self.YELLOW)
            self.screen.blit(title, (pad, pad))

            lines = [
                f"Pending labels: {summary.get('total', 0)}",
                f"Approved: {summary.get('approve', 0)}   Rejected: {summary.get('reject', 0)}   Review: {summary.get('review', 0)}",
            ]

            if suggestions:
                lines.append("Suggested updates:")
                for key, value in suggestions.items():
                    lines.append(f"  - {key}: {value:.4f}")
            else:
                lines.append("No threshold updates suggested yet (need stronger separation/more data).")

            if learned_range_lines:
                lines.append("Learned range updates:")
                lines.extend(learned_range_lines)

            if review_warnings:
                lines.append("Review warnings:")
                for warning in review_warnings:
                    lines.append(f"  - {warning}")

            lines.append("Choose action: Defer keeps collecting, Discard drops pending learning data, Update applies now.")

            y = pad + title.get_height() + 10
            line_height = self.small_font.get_linesize() + 3
            for line in lines:
                text = self.small_font.render(line, True, self.WHITE)
                self.screen.blit(text, (pad, y))
                y += line_height

            button_h = self._clamp(int(height * 0.08), 42, 64)
            button_w = self._clamp(int(width * 0.20), 140, 280)
            gap = self._clamp(int(width * 0.02), 10, 28)
            total_w = button_w * 4 + gap * 3
            start_x = (width - total_w) // 2
            btn_y = height - button_h - pad

            button_rects.clear()
            button_rects["defer"] = pygame.Rect(start_x, btn_y, button_w, button_h)
            button_rects["discard"] = pygame.Rect(start_x + (button_w + gap), btn_y, button_w, button_h)
            button_rects["update"] = pygame.Rect(start_x + (button_w + gap) * 2, btn_y, button_w, button_h)
            button_rects["home"] = pygame.Rect(start_x + (button_w + gap) * 3, btn_y, button_w, button_h)

            labels = {
                "defer": ("DEFER", defer_color),
                "discard": ("DISCARD", discard_color),
                "update": ("UPDATE", update_color),
                "home": ("HOME", home_color),
            }

            for key, rect in button_rects.items():
                text_label, color = labels[key]
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                text = self.small_font.render(text_label, True, self.WHITE)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)

            pygame.display.flip()

        render()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.type == pygame.VIDEORESIZE:
                    new_width = max(event.w, self.MIN_WIDTH)
                    new_height = max(event.h, self.MIN_HEIGHT)
                    self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                    render()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    for action, rect in button_rects.items():
                        if rect.collidepoint(pos):
                            return action

            self.clock.tick(30)


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

    def log_review_findings(self, warnings: list[str]):
        """Log review-stage warnings about config fit and prerequisites."""
        if warnings:
            self._log("=== REVIEW WARNINGS ===")
            for warning in warnings:
                self._log(warning)
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
            # Backward compatibility: old records had no learning_state.
            changed = False
            for record in self.training_data:
                if "learning_state" not in record:
                    record["learning_state"] = "committed"
                    changed = True
                if self._normalize_record_schema(record):
                    changed = True
            if changed:
                self.save_training_data()

    def save_training_data(self):
        """Save training data."""
        training_file = self.config_path.parent / "training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=2)

    def _load_current_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _default_final_class(feedback: str) -> str | None:
        normalized = str(feedback).strip().lower()
        if normalized == 'approve':
            return 'good'
        if normalized == 'reject':
            return 'reject'
        return None

    def _build_config_fingerprint(self, config: dict | None = None) -> dict:
        source = config or self._load_current_config()
        inspection_cfg = source.get('inspection', {}) if isinstance(source, dict) else {}
        alignment_cfg = source.get('alignment', {}) if isinstance(source, dict) else {}
        return {
            'inspection_mode': inspection_cfg.get('inspection_mode', 'mask_only'),
            'reference_strategy': inspection_cfg.get('reference_strategy', 'golden_only'),
            'blend_mode': inspection_cfg.get('blend_mode', 'hard_only'),
            'tolerance_mode': inspection_cfg.get('tolerance_mode', 'balanced'),
            'threshold_mode': inspection_cfg.get('threshold_mode'),
            'threshold_value': inspection_cfg.get('threshold_value'),
            'min_required_coverage': inspection_cfg.get('min_required_coverage'),
            'max_outside_allowed_ratio': inspection_cfg.get('max_outside_allowed_ratio'),
            'min_section_coverage': inspection_cfg.get('min_section_coverage'),
            'min_ssim': inspection_cfg.get('min_ssim'),
            'max_mse': inspection_cfg.get('max_mse'),
            'min_anomaly_score': inspection_cfg.get('min_anomaly_score'),
            'alignment_profile': alignment_cfg.get('tolerance_profile', 'balanced'),
        }

    def _normalize_record_schema(self, record: dict) -> bool:
        changed = False
        if record.get('schema_version') != TRAINING_DATA_SCHEMA_VERSION:
            record['schema_version'] = TRAINING_DATA_SCHEMA_VERSION
            changed = True
        if 'final_class' not in record:
            record['final_class'] = self._default_final_class(record.get('feedback', ''))
            changed = True
        if 'defect_category' not in record:
            record['defect_category'] = None
            changed = True
        if 'classification_reason' not in record:
            record['classification_reason'] = None
            changed = True
        if 'config_fingerprint' not in record:
            record['config_fingerprint'] = self._build_config_fingerprint()
            changed = True
        if 'record_id' not in record:
            record['record_id'] = f"legacy_{int(record.get('timestamp', time.time()) * 1000)}"
            changed = True
        if 'reference_candidate_id' not in record:
            record['reference_candidate_id'] = None
            changed = True
        if 'reference_candidate_state' not in record:
            record['reference_candidate_state'] = None
            changed = True
        return changed

    @staticmethod
    def _resolve_learning_class(record: dict) -> str | None:
        final_class = record.get('final_class')
        if final_class in {'good', 'reject'}:
            return final_class
        feedback = str(record.get('feedback', '')).strip().lower()
        if feedback == 'approve':
            return 'good'
        if feedback == 'reject':
            return 'reject'
        return None

    def record_feedback(
        self,
        details: dict,
        feedback: str,
        label_info: Optional[dict] = None,
        image_path: Optional[Path] = None,
    ):
        """Record human feedback for threshold adjustment."""
        label_info = label_info or {}
        final_class = label_info.get('final_class', self._default_final_class(feedback))
        record_id = f"feedback_{int(time.time() * 1000)}_{len(self.training_data) + 1}"
        record = {
            'schema_version': TRAINING_DATA_SCHEMA_VERSION,
            'record_id': record_id,
            'timestamp': time.time(),
            'feedback': feedback,
            'final_class': final_class,
            'defect_category': label_info.get('defect_category'),
            'classification_reason': label_info.get('classification_reason'),
            'learning_state': 'pending',
            'config_fingerprint': self._build_config_fingerprint(),
            'reference_candidate_id': None,
            'reference_candidate_state': None,
            'metrics': {
                'required_coverage': details.get('required_coverage', 0),
                'outside_allowed_ratio': details.get('outside_allowed_ratio', 0),
                'min_section_coverage': details.get('min_section_coverage', 0),
                'ssim': details.get('ssim'),
                'mse': details.get('mse'),
                'anomaly_score': details.get('anomaly_score'),
                'histogram_similarity': details.get('histogram_similarity'),
                'best_angle_deg': details.get('best_angle_deg', 0),
                'best_shift_x': details.get('best_shift_x', 0),
                'best_shift_y': details.get('best_shift_y', 0),
                'inspection_mode': details.get('inspection_mode', 'mask_only'),
            }
        }

        current_config = self._load_current_config()
        inspection_cfg = current_config.get('inspection', {})
        reference_strategy = str(inspection_cfg.get('reference_strategy', 'golden_only')).strip().lower()
        if (
            final_class == 'good'
            and image_path is not None
            and reference_strategy in {'hybrid', 'multi_good_experimental'}
        ):
            staged_ok, staged_result = stage_reference_candidate_from_image(
                current_config,
                image_path,
                label=f"Approved Good {len(self.training_data) + 1}",
                source_record_id=record_id,
            )
            if staged_ok:
                record['reference_candidate_id'] = staged_result['reference_id']
                record['reference_candidate_state'] = staged_result['state']
        self.training_data.append(record)
        self.save_training_data()

    def get_pending_summary(self) -> dict:
        """Get summary of pending (not yet committed/discarded) training records."""
        pending = self.get_pending_records()
        summary = {'approve': 0, 'reject': 0, 'review': 0, 'total': len(pending)}
        for record in pending:
            feedback = str(record.get('feedback', '')).lower()
            if feedback in summary:
                summary[feedback] += 1
        return summary

    def get_pending_records(self) -> list[dict]:
        return [r for r in self.training_data if r.get('learning_state', 'committed') == 'pending']

    @staticmethod
    def _round_learning_value(value) -> float:
        return round(float(value), 4)

    def extract_learned_ranges(self, records: Optional[list[dict]] = None) -> dict:
        source_records = records if records is not None else self.get_pending_records()
        good_records = [r for r in source_records if self._resolve_learning_class(r) == 'good']
        reject_records = [r for r in source_records if self._resolve_learning_class(r) == 'reject']
        learned_ranges: dict[str, dict] = {}

        for metric_name, direction in LEARNED_RANGE_DIRECTIONS.items():
            good_values = [r.get('metrics', {}).get(metric_name) for r in good_records]
            good_values = [float(value) for value in good_values if value is not None]
            if not good_values:
                continue

            learned_metric = {
                'direction': direction,
                'good_min': self._round_learning_value(min(good_values)),
                'good_max': self._round_learning_value(max(good_values)),
                'good_mean': self._round_learning_value(sum(good_values) / len(good_values)),
                'good_count': len(good_values),
            }

            reject_values = [r.get('metrics', {}).get(metric_name) for r in reject_records]
            reject_values = [float(value) for value in reject_values if value is not None]
            if reject_values:
                learned_metric.update(
                    {
                        'reject_min': self._round_learning_value(min(reject_values)),
                        'reject_max': self._round_learning_value(max(reject_values)),
                        'reject_mean': self._round_learning_value(sum(reject_values) / len(reject_values)),
                        'reject_count': len(reject_values),
                    }
                )

            learned_ranges[metric_name] = learned_metric

        return learned_ranges

    def summarize_learned_ranges(self, learned_ranges: dict) -> list[str]:
        lines: list[str] = []
        for metric_name in [
            'required_coverage',
            'outside_allowed_ratio',
            'min_section_coverage',
            'ssim',
            'mse',
            'anomaly_score',
        ]:
            learned_metric = learned_ranges.get(metric_name)
            if not learned_metric:
                continue
            lines.append(
                f"  - {metric_name}: good {learned_metric['good_min']:.4f} to {learned_metric['good_max']:.4f}"
            )
        return lines

    def _threshold_suggestion_from_range(
        self,
        learned_metric: dict,
        current_value,
        direction: str,
    ):
        suggested = None
        if direction == 'higher_is_better':
            good_floor = float(learned_metric['good_min'])
            reject_max = learned_metric.get('reject_max')
            if reject_max is not None and good_floor > float(reject_max):
                suggested = (good_floor + float(reject_max)) / 2.0
            elif current_value is None or float(current_value) > good_floor:
                suggested = good_floor
        else:
            good_ceiling = float(learned_metric['good_max'])
            reject_min = learned_metric.get('reject_min')
            if reject_min is not None and good_ceiling < float(reject_min):
                suggested = (good_ceiling + float(reject_min)) / 2.0
            elif current_value is None or float(current_value) < good_ceiling:
                suggested = good_ceiling

        if suggested is None:
            return None
        return self._round_learning_value(suggested)

    def _record_passes_current_thresholds(self, record: dict, inspection_cfg: dict) -> bool:
        metrics = record.get('metrics', {})
        required_coverage = float(metrics.get('required_coverage', 0.0))
        outside_allowed_ratio = float(metrics.get('outside_allowed_ratio', 1.0))
        min_section_coverage = float(metrics.get('min_section_coverage', 0.0))

        if required_coverage < float(inspection_cfg.get('min_required_coverage', 0.92)):
            return False
        if outside_allowed_ratio > float(inspection_cfg.get('max_outside_allowed_ratio', 0.02)):
            return False
        if min_section_coverage < float(inspection_cfg.get('min_section_coverage', 0.85)):
            return False

        min_ssim = inspection_cfg.get('min_ssim')
        if min_ssim not in {None, ''}:
            ssim_value = metrics.get('ssim')
            if ssim_value is None or float(ssim_value) < float(min_ssim):
                return False

        max_mse = inspection_cfg.get('max_mse')
        if max_mse not in {None, ''}:
            mse_value = metrics.get('mse')
            if mse_value is None or float(mse_value) > float(max_mse):
                return False

        min_anomaly_score = inspection_cfg.get('min_anomaly_score')
        if min_anomaly_score not in {None, ''}:
            anomaly_score = metrics.get('anomaly_score')
            if anomaly_score is None or float(anomaly_score) < float(min_anomaly_score):
                return False

        return True

    def get_training_review_warnings(
        self,
        config: dict,
        runtime_warnings: Optional[list[str]] = None,
        reference_warning: Optional[str] = None,
    ) -> list[str]:
        warnings: list[str] = []
        if reference_warning:
            warnings.append(reference_warning)
        for warning in runtime_warnings or []:
            if warning not in warnings:
                warnings.append(warning)

        pending = self.get_pending_records()
        if not pending:
            return warnings

        current_fingerprint = self._build_config_fingerprint(config)
        config_mismatch_count = sum(1 for record in pending if record.get('config_fingerprint') != current_fingerprint)
        if config_mismatch_count:
            warnings.append(
                f"{config_mismatch_count} pending examples were collected under different config settings than the current project config."
            )

        inspection_cfg = config.get('inspection', {})
        good_records = [r for r in pending if self._resolve_learning_class(r) == 'good']
        reject_records = [r for r in pending if self._resolve_learning_class(r) == 'reject']

        approved_failures = sum(1 for record in good_records if not self._record_passes_current_thresholds(record, inspection_cfg))
        if approved_failures:
            warnings.append(
                f"{approved_failures} approved-good pending examples fail the current thresholds. Config may be too strict for this project."
            )

        reject_passes = sum(1 for record in reject_records if self._record_passes_current_thresholds(record, inspection_cfg))
        if reject_passes:
            warnings.append(
                f"{reject_passes} reject-labeled pending examples still pass the current thresholds. Config may be too loose or the reference may not fit the project."
            )

        return warnings

    def commit_pending_feedback(self) -> int:
        """Mark pending records as committed after applying threshold updates."""
        updated = 0
        for record in self.training_data:
            if record.get('learning_state', 'committed') == 'pending':
                candidate_id = record.get('reference_candidate_id')
                candidate_state = record.get('reference_candidate_state')
                learning_class = self._resolve_learning_class(record)
                if candidate_id and candidate_state == 'pending' and learning_class == 'good':
                    if not activate_reference_candidate(candidate_id):
                        continue
                    record['reference_candidate_state'] = 'active'
                record['learning_state'] = 'committed'
                updated += 1
        if updated:
            self.save_training_data()
        return updated

    def discard_pending_feedback(self) -> int:
        """Exclude pending records from learning while preserving audit history."""
        updated = 0
        for record in self.training_data:
            if record.get('learning_state', 'committed') == 'pending':
                candidate_id = record.get('reference_candidate_id')
                candidate_state = record.get('reference_candidate_state')
                if candidate_id and candidate_state == 'pending':
                    discard_reference_candidate(candidate_id, state='pending')
                    record['reference_candidate_state'] = 'discarded'
                record['learning_state'] = 'discarded'
                updated += 1
        if updated:
            self.save_training_data()
        return updated

    def suggest_thresholds(self) -> dict:
        """Analyze training data and suggest new thresholds."""
        learning_records = self.get_pending_records()
        if len(learning_records) < 10:
            return {}  # Need more data

        learned_ranges = self.extract_learned_ranges(learning_records)
        if not learned_ranges:
            return {}

        inspection_cfg = self._load_current_config().get('inspection', {})
        inspection_mode = normalize_inspection_mode(inspection_cfg.get('inspection_mode', 'mask_only'))
        suggestions = {}

        metric_map = [
            ('required_coverage', 'min_required_coverage'),
            ('outside_allowed_ratio', 'max_outside_allowed_ratio'),
            ('min_section_coverage', 'min_section_coverage'),
        ]
        for metric_name, config_key in metric_map:
            learned_metric = learned_ranges.get(metric_name)
            if not learned_metric:
                continue
            suggestion = self._threshold_suggestion_from_range(
                learned_metric,
                inspection_cfg.get(config_key),
                LEARNED_RANGE_DIRECTIONS[metric_name],
            )
            if suggestion is not None:
                suggestions[config_key] = suggestion

        optional_metric_map = [
            ('ssim', 'min_ssim'),
            ('mse', 'max_mse'),
            ('anomaly_score', 'min_anomaly_score'),
        ]
        active_optional_metrics = INSPECTION_MODE_OPTIONAL_METRICS[inspection_mode]
        for metric_name, config_key in optional_metric_map:
            learned_metric = learned_ranges.get(metric_name)
            if not learned_metric:
                continue
            if metric_name not in active_optional_metrics:
                continue
            suggestion = self._threshold_suggestion_from_range(
                learned_metric,
                inspection_cfg.get(config_key),
                LEARNED_RANGE_DIRECTIONS[metric_name],
            )
            if suggestion is not None:
                suggestions[config_key] = suggestion

        if self.logger and suggestions:
            self.logger.log_threshold_suggestion(suggestions)

        return suggestions

    def apply_learning_update(self, config: dict, suggestions: dict, learned_ranges: dict) -> dict:
        threshold_updates = self.apply_suggestions(config, suggestions)

        inspection_cfg = config.setdefault('inspection', {})
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
        else:
            file_config = config.copy()

        file_inspection_cfg = file_config.setdefault('inspection', {})
        learned_ranges_changed = file_inspection_cfg.get('learned_ranges') != learned_ranges
        if learned_ranges:
            inspection_cfg['learned_ranges'] = learned_ranges
            file_inspection_cfg['learned_ranges'] = learned_ranges

        if learned_ranges_changed and learned_ranges:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(file_config, f, indent=2)
                f.write('\n')

        return {
            'threshold_updates': threshold_updates,
            'learned_ranges_saved': bool(learned_ranges),
            'learned_ranges_changed': learned_ranges_changed and bool(learned_ranges),
        }

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
            current_value = inspection_cfg.get(key)
            if current_value not in {None, ''} and float(current_value) == normalized_value:
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
        clear_reference_variants(active_paths)
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
    reference_candidates = list_runtime_reference_candidates(config, active_paths)

    anomaly_detector = load_anomaly_detector(active_paths)
    for warning in get_inspection_runtime_warnings(config, anomaly_detector):
        print(f"Warning: {warning}")

    # Initialize logger
    log_dir = active_paths["log_dir"]
    logger = TrainingLogger(log_dir)
    logger.start_session()

    display = InspectionDisplay()
    trainer = ThresholdTrainer(active_paths["config_file"], logger)
    display.set_alignment_profile_label(config.get("alignment", {}).get("tolerance_profile", "balanced"))
    mode_warnings = get_inspection_runtime_warnings(config, anomaly_detector)

    def build_review_warnings() -> list[str]:
        settings_match, reference_warning = check_reference_settings_match(config)
        if settings_match:
            reference_warning = None
        warnings = trainer.get_training_review_warnings(
            config,
            runtime_warnings=mode_warnings,
            reference_warning=reference_warning,
        )
        return warnings

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

    def should_review_training(count: int) -> bool:
        if count <= 0:
            return False
        if count <= early_review_parts:
            return count % early_review_interval == 0
        return count % steady_review_interval == 0

    try:
        print("Starting interactive training mode...")
        print("Controls:")
        print("- Green APPROVE button: Accept the sample")
        print("- Red REJECT button: Reject the sample")
        print("- Yellow REVIEW button: Flag for human review")
        print("- Blue SET REF button: Capture a new golden reference")
        print("- Close window or Esc to exit")
        print("\nDetailed descriptions will appear on screen for each inspection.")
        initial_review_warnings = build_review_warnings()
        if initial_review_warnings:
            logger.log_review_findings(initial_review_warnings)
            for warning in initial_review_warnings:
                print(f"Warning: {warning}")
            display.show_message(initial_review_warnings[0], display.YELLOW)
            time.sleep(1.2)

        session_count = 0

        # If no reference exists yet, show live preview until the operator captures one.
        if not reference_candidates:
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
                reference_candidates = list_runtime_reference_candidates(config, active_paths)
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
                    reference_candidates = list_runtime_reference_candidates(config, active_paths)
                    passed, details = inspect_against_references(
                        config,
                        image_path,
                        reference_candidates,
                        make_binary_mask,
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
                feedback, ready_to_capture = display.display_inspection(
                    image_path,
                    passed,
                    details,
                    logger,
                    config=config,
                    reference_mask_path=active_paths["reference_mask"],
                )

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
                elif feedback in ['approve', 'reject', 'review']:
                    if not ready_to_capture:
                        # This shouldn't happen in normal flow; log and continue
                        print(f"Warning: Got feedback but not ready to capture. Continuing...")
                        continue
                    
                    trainer.record_feedback(details, feedback, image_path=image_path)
                    session_count += 1

                    # Update indicators
                    if feedback == 'approve':
                        indicator.pulse_pass()
                    elif feedback == 'reject':
                        indicator.pulse_fail()
                    # No indicator for review

                    # Show session summary at configured review cadence.
                    if should_review_training(session_count):
                        summary = trainer.get_pending_summary()
                        suggestions = trainer.suggest_thresholds()
                        learned_ranges = trainer.extract_learned_ranges()
                        learned_range_lines = trainer.summarize_learned_ranges(learned_ranges)
                        review_warnings = build_review_warnings()
                        if review_warnings:
                            logger.log_review_findings(review_warnings)
                        action = display.show_training_checkpoint(
                            summary,
                            suggestions,
                            review_warnings,
                            learned_range_lines,
                        )
                        if action in {'quit', 'home'}:
                            print("Returning to dashboard/home.")
                            break
                        if action == 'defer':
                            print("Deferred updates; continuing to collect training data.")
                        elif action == 'discard':
                            discarded = trainer.discard_pending_feedback()
                            print(f"Discarded {discarded} pending training records.")
                        elif action == 'update':
                            if not suggestions and not learned_ranges:
                                print("No learned updates to apply yet; continuing with current settings.")
                            else:
                                applied = trainer.apply_learning_update(config, suggestions, learned_ranges)
                                if applied['threshold_updates'] or applied['learned_ranges_changed']:
                                    committed = trainer.commit_pending_feedback()
                                    if applied['threshold_updates']:
                                        print("Applied threshold updates:")
                                    for key, value in applied['threshold_updates'].items():
                                        print(f"  {key}: {value:.4f}")
                                    if applied['learned_ranges_saved']:
                                        print(f"Saved learned ranges for {len(learned_ranges)} metrics.")
                                    print(f"Committed {committed} training records.")
                                else:
                                    print("No learning changes were applied.")

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

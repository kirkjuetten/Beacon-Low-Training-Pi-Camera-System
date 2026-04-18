#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import pygame
    import pygame.gfxdraw

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from inspection_system.app.anomaly_detection_utils import AnomalyDetector
from inspection_system.app.alignment_utils import align_sample_mask
from inspection_system.app.camera_interface import get_active_runtime_paths, import_cv2_and_numpy
from inspection_system.app.frame_acquisition import capture_to_temp, cleanup_temp_image
from inspection_system.app.inspection_runtime_context import build_inspection_runtime_context
from inspection_system.app.inspection_pipeline import inspect_against_references
from inspection_system.app.morphology_utils import dilate_mask, erode_mask
from inspection_system.app.preprocessing_utils import make_binary_mask
from inspection_system.app.reference_region_utils import build_reference_regions
from inspection_system.app.reference_service import list_runtime_reference_candidates, save_debug_outputs
from inspection_system.app.result_interpreter import (
    GOOD,
    REASON_EXTRA_PRINT,
    REASON_REGISTRATION_FAILURE,
    REASON_LABELS,
    REASON_MISSING_PRINT,
    REASON_ORDER,
    REASON_REFERENCE_MISMATCH,
    REASON_UNEVEN_PRINT,
    REJECT,
    REVIEW,
    ProductionOutcome,
    determine_operator_outcome,
)
from inspection_system.app.runtime_controller import (
    format_operator_mode_lines,
    get_inspection_runtime_warnings,
    load_anomaly_detector,
)
from inspection_system.app.scoring_utils import evaluate_metrics, score_sample
from inspection_system.app.section_mask_utils import compute_section_masks


@dataclass
class CounterScope:
    total: int = 0
    good: int = 0
    reject: int = 0
    review: int = 0
    reject_reasons: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in REASON_ORDER}
    )

    def record(self, outcome: ProductionOutcome) -> None:
        self.total += 1
        if outcome.status == GOOD:
            self.good += 1
            return
        if outcome.status == REVIEW:
            self.review += 1
            return

        self.reject += 1
        if outcome.primary_reason:
            self.reject_reasons[outcome.primary_reason] = self.reject_reasons.get(outcome.primary_reason, 0) + 1

    def reset(self) -> None:
        self.total = 0
        self.good = 0
        self.reject = 0
        self.review = 0
        self.reject_reasons = {reason: 0 for reason in REASON_ORDER}


@dataclass
class ProductionSessionState:
    run_totals: CounterScope = field(default_factory=CounterScope)
    shift_totals: CounterScope = field(default_factory=CounterScope)
    last_outcome: Optional[ProductionOutcome] = None
    last_details: dict = field(default_factory=dict)
    display_mode: str = "raw"

    def record(self, outcome: ProductionOutcome, details: dict) -> None:
        self.run_totals.record(outcome)
        self.shift_totals.record(outcome)
        self.last_outcome = outcome
        self.last_details = details

class ProductionDisplay:
    MIN_WIDTH = 640
    MIN_HEIGHT = 420

    def __init__(self, width: int = 1280, height: int = 800):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for production display. Install with: pip install pygame")

        pygame.init()
        initial_width, initial_height = self.clamp_window_size(width, height)

        self.screen = pygame.display.set_mode((initial_width, initial_height), pygame.RESIZABLE)
        pygame.display.set_caption("Beacon Production Inspection")
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        self.large_font = pygame.font.Font(None, 58)
        self.clock = pygame.time.Clock()

        self.GREEN = (28, 176, 73)
        self.RED = (206, 39, 45)
        self.YELLOW = (242, 170, 38)
        self.WHITE = (245, 245, 245)
        self.BLACK = (16, 20, 24)
        self.GRAY = (99, 109, 124)
        self.SLATE = (34, 42, 52)
        self.BLUE = (41, 98, 255)

        self.buttons: dict[str, pygame.Rect] = {}
        self.layout: dict[str, pygame.Rect] = {}
        self._reflow_layout()

    def cleanup(self) -> None:
        try:
            pygame.quit()
        except Exception:
            pass

    @staticmethod
    def _clamp(value: int, lower: int, upper: int) -> int:
        return max(lower, min(value, upper))

    def _window_bounds(self) -> tuple[int, int, int, int]:
        display_info = pygame.display.Info()
        max_width = max(480, display_info.current_w - 24)
        max_height = max(360, display_info.current_h - 56)
        min_width = min(self.MIN_WIDTH, max_width)
        min_height = min(self.MIN_HEIGHT, max_height)
        return min_width, min_height, max_width, max_height

    def clamp_window_size(self, width: int, height: int) -> tuple[int, int]:
        min_width, min_height, max_width, max_height = self._window_bounds()
        clamped_width = max(min_width, min(width, max_width))
        clamped_height = max(min_height, min(height, max_height))
        return clamped_width, clamped_height

    def _update_fonts(self) -> None:
        _, height = self.screen.get_size()
        base_size = self._clamp(int(height * 0.035), 18, 34)
        small_size = self._clamp(int(height * 0.026), 14, 24)
        large_size = self._clamp(int(height * 0.07), 36, 72)
        self.font = pygame.font.Font(None, base_size)
        self.small_font = pygame.font.Font(None, small_size)
        self.large_font = pygame.font.Font(None, large_size)

    def _reflow_layout(self) -> None:
        width, height = self.screen.get_size()
        pad = self._clamp(int(height * 0.018), 6, 20)
        banner_h = self._clamp(int(height * 0.13), 54, 120)
        controls_h = self._clamp(int(height * 0.12), 58, 120)
        content_h = height - banner_h - controls_h - pad * 4
        sidebar_w = self._clamp(int(width * 0.29), 200, 430)
        detail_h = self._clamp(int(content_h * 0.24), 84, 210)

        banner_rect = pygame.Rect(pad, pad, width - pad * 2, banner_h)
        sidebar_rect = pygame.Rect(pad, banner_rect.bottom + pad, sidebar_w, content_h)
        image_rect = pygame.Rect(
            sidebar_rect.right + pad,
            banner_rect.bottom + pad,
            width - sidebar_w - pad * 3,
            content_h - detail_h - pad,
        )
        detail_rect = pygame.Rect(sidebar_rect.right + pad, image_rect.bottom + pad, image_rect.width, detail_h)
        controls_rect = pygame.Rect(pad, height - controls_h - pad, width - pad * 2, controls_h)

        self.layout = {
            "banner_rect": banner_rect,
            "sidebar_rect": sidebar_rect,
            "image_rect": image_rect,
            "detail_rect": detail_rect,
            "controls_rect": controls_rect,
        }

        self._update_fonts()
        self._layout_buttons(controls_rect)

    def _layout_buttons(self, controls_rect: pygame.Rect) -> None:
        gap = self._clamp(int(controls_rect.width * 0.015), 10, 22)
        pad = self._clamp(int(controls_rect.height * 0.16), 6, 18)
        button_h = controls_rect.height - pad * 2
        button_defs = [
            ("manual_inspect", 0.42),
            ("reset_run", 0.19),
            ("reset_shift", 0.19),
            ("home", 0.20),
        ]
        usable_w = controls_rect.width - gap * (len(button_defs) - 1)
        widths = [max(110, int(usable_w * ratio)) for _key, ratio in button_defs]
        widths[-1] = max(100, controls_rect.width - sum(widths[:-1]) - gap * (len(button_defs) - 1))

        x = controls_rect.x
        y = controls_rect.y + pad
        self.buttons = {}
        for (key, _ratio), width in zip(button_defs, widths):
            self.buttons[key] = pygame.Rect(x, y, width, button_h)
            x += width + gap

    def draw_image_with_border(
        self,
        surface: pygame.Surface,
        border_color: tuple[int, int, int],
        image_rect: pygame.Rect,
        border_width: int = 5,
    ) -> None:
        draw_rect = surface.get_rect()
        draw_rect.center = image_rect.center
        draw_rect.clamp_ip(image_rect)
        border_rect = draw_rect.inflate(border_width * 2, border_width * 2)
        pygame.draw.rect(self.screen, border_color, border_rect, border_width)
        self.screen.blit(surface, draw_rect.topleft)

    def _scale_surface_to_rect(self, surface: pygame.Surface, target_rect: pygame.Rect) -> pygame.Surface:
        img_width, img_height = surface.get_size()
        available_w = max(1, target_rect.width - 10)
        available_h = max(1, target_rect.height - 10)
        scale = min(available_w / img_width, available_h / img_height)
        scale = min(scale, 2.4)
        new_width = max(1, int(img_width * scale))
        new_height = max(1, int(img_height * scale))
        return pygame.transform.smoothscale(surface, (new_width, new_height))

    def load_surface_from_image(self, image_path: Path) -> Optional[pygame.Surface]:
        cv2, _ = import_cv2_and_numpy()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(image.swapaxes(0, 1))

    def _make_processed_surface(self, image_path: Path, config: dict) -> Optional[pygame.Surface]:
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
                0,
                255,
            ).astype(np.uint8)
            return pygame.surfarray.make_surface(overlay.swapaxes(0, 1))
        except Exception:
            return None

    def show_message(self, message: str, color: tuple[int, int, int]) -> None:
        self._reflow_layout()
        self.screen.fill(self.BLACK)
        text = self.large_font.render(message, True, color)
        text_rect = text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

    def _wrap_lines(self, text: str, font: pygame.font.Font, width: int) -> list[str]:
        words = text.split()
        if not words:
            return []
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _draw_scope(self, rect: pygame.Rect, title: str, scope: CounterScope) -> None:
        pygame.draw.rect(self.screen, self.SLATE, rect, border_radius=10)
        pygame.draw.rect(self.screen, self.GRAY, rect, 2, border_radius=10)
        title_surface = self.font.render(title, True, self.WHITE)
        self.screen.blit(title_surface, (rect.x + 14, rect.y + 12))

        compact = rect.height < 170
        y = rect.y + title_surface.get_height() + 16
        line_height = self.small_font.get_linesize() + 3

        if compact:
            totals_line = f"T:{scope.total}  G:{scope.good}  R:{scope.reject}  V:{scope.review}"
            totals_surface = self.small_font.render(totals_line, True, self.WHITE)
            self.screen.blit(totals_surface, (rect.x + 14, y))
            y += line_height + 2

            compact_reason_labels = {
                REASON_MISSING_PRINT: "Miss",
                REASON_EXTRA_PRINT: "Extra",
                REASON_UNEVEN_PRINT: "Uneven",
                REASON_REFERENCE_MISMATCH: "Match",
                REASON_REGISTRATION_FAILURE: "Place",
            }
            for index in range(0, len(REASON_ORDER), 2):
                reasons = REASON_ORDER[index : index + 2]
                row = "  ".join(
                    f"{compact_reason_labels.get(reason, reason[:5])}:{scope.reject_reasons.get(reason, 0)}"
                    for reason in reasons
                )
                line = self.small_font.render(row, True, self.WHITE)
                self.screen.blit(line, (rect.x + 14, y))
                y += line_height
            return

        rows = [
            f"Total: {scope.total}",
            f"Good: {scope.good}",
            f"Reject: {scope.reject}",
            f"Review: {scope.review}",
        ]
        for row in rows:
            line = self.small_font.render(row, True, self.WHITE)
            self.screen.blit(line, (rect.x + 16, y))
            y += line_height

        y += 4
        subtitle = self.small_font.render("Reject reasons", True, self.YELLOW)
        self.screen.blit(subtitle, (rect.x + 16, y))
        y += line_height
        for reason in REASON_ORDER:
            label = f"{REASON_LABELS[reason]}: {scope.reject_reasons.get(reason, 0)}"
            line = self.small_font.render(label, True, self.WHITE)
            self.screen.blit(line, (rect.x + 16, y))
            y += line_height

    def _build_display_surface(
        self,
        source_surface: Optional[pygame.Surface],
        processed_surface: Optional[pygame.Surface],
        display_mode: str,
        image_rect: pygame.Rect,
    ) -> Optional[pygame.Surface]:
        if source_surface is None:
            return None
        if display_mode == "processed" and processed_surface is not None:
            return self._scale_surface_to_rect(processed_surface, image_rect)
        if display_mode == "split" and processed_surface is not None:
            half_w = max(1, (image_rect.width - 8) // 2)
            split_h = max(1, image_rect.height - 10)
            left = pygame.transform.smoothscale(source_surface, (half_w, split_h))
            right = pygame.transform.smoothscale(processed_surface, (half_w, split_h))
            combined = pygame.Surface((half_w * 2 + 8, split_h))
            combined.fill((48, 48, 48))
            combined.blit(left, (0, 0))
            combined.blit(right, (half_w + 8, 0))
            return combined
        return self._scale_surface_to_rect(source_surface, image_rect)

    def render(
        self,
        session: ProductionSessionState,
        source_surface: Optional[pygame.Surface],
        processed_surface: Optional[pygame.Surface],
        status_message: str,
        mode_lines: Optional[list[str]] = None,
    ) -> None:
        self._reflow_layout()
        self.screen.fill(self.BLACK)

        banner_rect = self.layout["banner_rect"]
        sidebar_rect = self.layout["sidebar_rect"]
        image_rect = self.layout["image_rect"]
        detail_rect = self.layout["detail_rect"]

        outcome = session.last_outcome
        if outcome is None:
            banner_color = self.BLUE
            banner_text = "READY FOR INSPECTION"
        elif outcome.status == GOOD:
            banner_color = self.GREEN
            banner_text = outcome.banner_text
        elif outcome.status == REVIEW:
            banner_color = self.YELLOW
            banner_text = outcome.banner_text
        else:
            banner_color = self.RED
            banner_text = outcome.banner_text

        pygame.draw.rect(self.screen, banner_color, banner_rect, border_radius=12)
        banner_surface = self.large_font.render(banner_text, True, self.WHITE)
        banner_text_rect = banner_surface.get_rect(center=(banner_rect.centerx, banner_rect.centery - 10))
        self.screen.blit(banner_surface, banner_text_rect)

        status_surface = self.small_font.render(status_message, True, self.WHITE)
        status_rect = status_surface.get_rect(center=(banner_rect.centerx, banner_rect.bottom - 22))
        self.screen.blit(status_surface, status_rect)

        scope_gap = 12
        scope_h = (sidebar_rect.height - scope_gap) // 2
        run_rect = pygame.Rect(sidebar_rect.x, sidebar_rect.y, sidebar_rect.width, scope_h)
        shift_rect = pygame.Rect(sidebar_rect.x, sidebar_rect.y + scope_h + scope_gap, sidebar_rect.width, scope_h)
        self._draw_scope(run_rect, "Run Totals", session.run_totals)
        self._draw_scope(shift_rect, "Shift Totals", session.shift_totals)

        display_surface = self._build_display_surface(source_surface, processed_surface, session.display_mode, image_rect)
        if display_surface is not None:
            self.draw_image_with_border(display_surface, banner_color, image_rect)
        else:
            placeholder = self.font.render("Press MANUAL INSPECT to check the next part.", True, self.WHITE)
            placeholder_rect = placeholder.get_rect(center=image_rect.center)
            self.screen.blit(placeholder, placeholder_rect)

        pygame.draw.rect(self.screen, self.SLATE, detail_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.GRAY, detail_rect, 2, border_radius=10)
        detail_lines = ["No part inspected yet."] if outcome is None else outcome.summary_lines
        y = detail_rect.y + 16
        for line in detail_lines:
            wrapped = self._wrap_lines(line, self.font, detail_rect.width - 24)
            for wrapped_line in wrapped:
                line_surface = self.font.render(wrapped_line, True, self.WHITE)
                self.screen.blit(line_surface, (detail_rect.x + 14, y))
                y += self.font.get_linesize() + 4

        if mode_lines:
            y += 6
            for line in mode_lines:
                wrapped = self._wrap_lines(line, self.small_font, detail_rect.width - 24)
                for wrapped_line in wrapped:
                    line_surface = self.small_font.render(wrapped_line, True, self.YELLOW)
                    self.screen.blit(line_surface, (detail_rect.x + 14, y))
                    y += self.small_font.get_linesize() + 2

        self.draw_buttons()
        pygame.display.flip()

    def draw_buttons(self) -> None:
        labels = {
            "manual_inspect": ("MANUAL INSPECT", self.BLUE),
            "reset_run": ("RESET RUN", self.GRAY),
            "reset_shift": ("RESET SHIFT", self.GRAY),
            "home": ("HOME", self.GRAY),
        }
        for key, rect in self.buttons.items():
            label, color = labels[key]
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, self.WHITE, rect, 2, border_radius=8)
            font = self.font if rect.width > 180 else self.small_font
            text = font.render(label, True, self.WHITE)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)


def _perform_inspection(
    config: dict,
    runtime_context,
) -> tuple[Optional[ProductionOutcome], dict, Optional[Path], Optional[str]]:
    result_code, image_path, stderr_text = capture_to_temp(config)
    if result_code != 0:
        cleanup_temp_image()
        return None, {}, None, stderr_text or "Capture failed."

    try:
        reference_candidates = runtime_context.reference_candidates
        if not reference_candidates:
            return None, {}, None, "No active runtime references are available. Capture a golden reference first."
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
            anomaly_detector=runtime_context.anomaly_detector,
        )
        return determine_operator_outcome(passed, details), details, image_path, None
    except Exception as exc:
        cleanup_temp_image()
        return None, {}, None, str(exc)


def run_production_mode(config: dict, indicator) -> int:
    if not PYGAME_AVAILABLE:
        print("pygame is required for production mode. Install with: pip install pygame")
        return 1

    runtime_context = build_inspection_runtime_context(
        config,
        active_paths_loader=get_active_runtime_paths,
        reference_candidates_loader=list_runtime_reference_candidates,
        anomaly_detector_loader=load_anomaly_detector,
    )
    active_paths = runtime_context.active_paths
    if not runtime_context.reference_candidates:
        print("Production mode requires an active reference image and mask.")
        print("Capture a reference first from the dashboard.")
        return 1

    display = ProductionDisplay()
    session = ProductionSessionState()
    source_surface: Optional[pygame.Surface] = None
    processed_surface: Optional[pygame.Surface] = None
    status_message = "Ready. Press MANUAL INSPECT to inspect the next part."
    session.display_mode = str(config.get("inspection", {}).get("image_display_mode", "raw")).strip().lower() or "raw"
    mode_warnings = get_inspection_runtime_warnings(config, runtime_context.anomaly_detector, active_paths)
    mode_lines = format_operator_mode_lines(config, active_paths, runtime_context.anomaly_detector)
    if mode_warnings:
        for warning in mode_warnings:
            print(f"Warning: {warning}")
        status_message = f"Warning: {mode_warnings[0]}"

    try:
        display.render(session, source_surface, processed_surface, status_message, mode_lines)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return 0
                if event.type == pygame.VIDEORESIZE:
                    new_width, new_height = display.clamp_window_size(event.w, event.h)
                    display.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                    display.render(session, source_surface, processed_surface, status_message, mode_lines)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if display.buttons["manual_inspect"].collidepoint(pos):
                        display.show_message("INSPECTING...", display.YELLOW)
                        outcome, details, image_path, error_message = _perform_inspection(config, runtime_context)
                        if error_message is not None:
                            source_surface = None
                            processed_surface = None
                            status_message = f"Inspection error: {error_message}"
                            display.render(session, source_surface, processed_surface, status_message, mode_lines)
                            continue

                        assert outcome is not None
                        assert image_path is not None
                        source_surface = display.load_surface_from_image(image_path)
                        processed_surface = None
                        if session.display_mode in {"processed", "split"}:
                            processed_surface = display._make_processed_surface(image_path, config)
                            if processed_surface is None:
                                session.display_mode = "raw"

                        session.record(outcome, details)
                        if outcome.status == GOOD:
                            indicator.pulse_pass()
                        elif outcome.status == REJECT:
                            indicator.pulse_fail()

                        reason_label = REASON_LABELS.get(outcome.primary_reason, "-") if outcome.primary_reason else "-"
                        status_message = f"Last result: {outcome.banner_text} | Reason: {reason_label}"
                        cleanup_temp_image()
                        display.render(session, source_surface, processed_surface, status_message, mode_lines)
                    elif display.buttons["reset_run"].collidepoint(pos):
                        session.run_totals.reset()
                        status_message = "Run totals reset. Shift totals preserved."
                        display.render(session, source_surface, processed_surface, status_message, mode_lines)
                    elif display.buttons["reset_shift"].collidepoint(pos):
                        session.shift_totals.reset()
                        status_message = "Shift totals reset. Run totals preserved."
                        display.render(session, source_surface, processed_surface, status_message, mode_lines)
                    elif display.buttons["home"].collidepoint(pos):
                        return 0

            display.clock.tick(30)
    finally:
        cleanup_temp_image()
        display.cleanup()
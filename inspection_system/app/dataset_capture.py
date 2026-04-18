#!/usr/bin/env python3
"""Guided capture workflow for project-scoped test datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageTk
    PIL_CAPTURE_PREVIEW = True
except ImportError:
    Image = None
    ImageTk = None
    PIL_CAPTURE_PREVIEW = False

from inspection_system.app.camera_interface import get_current_project
from inspection_system.app.frame_acquisition import capture_to_temp
from inspection_system.app.replay_inspection import inspect_file
from inspection_system.app.result_status import FAIL, INVALID_CAPTURE, PASS


DATASET_CAPTURE_SCHEMA_VERSION = 1
CAPTURE_DATASET_SPLITS = ["tuning", "validation", "regression"]
CAPTURE_BUCKETS = ["good", "reject", "borderline", "invalid_capture"]
COMPACT_CAPTURE_MAX_WIDTH = 1100
COMPACT_CAPTURE_MAX_HEIGHT = 700
DEFECT_CATEGORY_OPTIONS = [
    "",
    "light_pipe_position",
    "broken_coring",
    "short_shot",
    "flash",
    "missing_feature",
    "damaged_feature",
    "wrong_part",
    "no_part",
    "bad_seat",
    "shifted_load",
    "rotated_load",
    "glare",
    "blur",
    "clipped_frame",
    "contamination",
    "unknown",
]


@dataclass(frozen=True)
class CaptureSession:
    project_id: str
    part_id: str
    camera_setup_id: str
    dataset_split: str
    session_label: str
    session_id: str
    session_dir: Path
    images_dir: Path
    manifest_path: Path
    session_metadata_path: Path
    auto_replay_after_capture: bool
    inspection_mode: str
    reference_strategy: str
    config_fingerprint: str
    config_file: Path
    reference_dir: Path
    roi_snapshot: dict
    capture_settings: dict


def slugify_identifier(value: str | None, fallback: str = "default") -> str:
    text = str(value or "").strip().lower()
    slug = []
    previous_dash = False
    for char in text:
        if char.isalnum():
            slug.append(char)
            previous_dash = False
        elif not previous_dash:
            slug.append("-")
            previous_dash = True
    normalized = "".join(slug).strip("-")
    return normalized or fallback


def compute_config_fingerprint(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def format_display_path(path: str | Path, *, keep_parts: int = 3) -> str:
    resolved = Path(path)
    parts = list(resolved.parts)
    if len(parts) <= keep_parts:
        return str(resolved)
    return ".../" + "/".join(parts[-keep_parts:])


def should_use_compact_capture_layout(screen_width: int, screen_height: int) -> bool:
    return screen_width < COMPACT_CAPTURE_MAX_WIDTH or screen_height < COMPACT_CAPTURE_MAX_HEIGHT


def get_project_root(active_paths: dict) -> Path:
    config_file = Path(active_paths["config_file"])
    parent = config_file.parent
    if parent.name == "config":
        return parent.parent
    return parent


def get_test_data_root(active_paths: dict) -> Path:
    root = get_project_root(active_paths) / "test_data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def expected_runtime_status(bucket: str) -> str | None:
    mapping = {
        "good": PASS,
        "reject": FAIL,
        "invalid_capture": INVALID_CAPTURE,
    }
    return mapping.get(bucket)


def build_collection_defaults(config: dict, active_paths: dict, project_name: str | None = None) -> dict:
    dataset_cfg = config.get("dataset_capture", {})
    inspection_cfg = config.get("inspection", {})
    capture_cfg = config.get("capture", {})

    resolved_project = str(project_name or get_current_project() or "default").strip() or "default"
    part_id = str(dataset_cfg.get("part_id") or resolved_project).strip() or resolved_project
    camera_setup_id = str(dataset_cfg.get("camera_setup_id") or f"{resolved_project}_default").strip() or f"{resolved_project}_default"
    default_split = str(dataset_cfg.get("default_split") or "tuning").strip().lower() or "tuning"
    if default_split not in CAPTURE_DATASET_SPLITS:
        default_split = "tuning"

    return {
        "project_id": resolved_project,
        "part_id": part_id,
        "camera_setup_id": camera_setup_id,
        "default_split": default_split,
        "auto_replay_after_capture": bool(dataset_cfg.get("auto_replay_after_capture", True)),
        "inspection_mode": str(inspection_cfg.get("inspection_mode", "mask_only")),
        "reference_strategy": str(inspection_cfg.get("reference_strategy", "golden_only")),
        "config_file": str(active_paths["config_file"]),
        "reference_dir": str(active_paths["reference_dir"]),
        "roi_snapshot": json.loads(json.dumps(inspection_cfg.get("roi", {}))),
        "capture_settings": json.loads(json.dumps(capture_cfg)),
    }


def persist_collection_defaults(
    config_file: Path,
    config: dict,
    *,
    part_id: str,
    camera_setup_id: str,
    default_split: str,
    auto_replay_after_capture: bool,
) -> None:
    updated = json.loads(json.dumps(config))
    dataset_cfg = updated.setdefault("dataset_capture", {})
    dataset_cfg["part_id"] = str(part_id).strip() or None
    dataset_cfg["camera_setup_id"] = str(camera_setup_id).strip() or None
    dataset_cfg["default_split"] = default_split if default_split in CAPTURE_DATASET_SPLITS else "tuning"
    dataset_cfg["auto_replay_after_capture"] = bool(auto_replay_after_capture)
    config_file.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")


def create_capture_session(config: dict, active_paths: dict, project_name: str | None, form_data: dict) -> CaptureSession:
    defaults = build_collection_defaults(config, active_paths, project_name)
    project_id = str(project_name or defaults["project_id"] or "default").strip() or "default"
    part_id = str(form_data.get("part_id") or defaults["part_id"]).strip() or project_id
    camera_setup_id = str(form_data.get("camera_setup_id") or defaults["camera_setup_id"]).strip() or f"{project_id}_default"
    dataset_split = str(form_data.get("dataset_split") or defaults["default_split"]).strip().lower() or "tuning"
    if dataset_split not in CAPTURE_DATASET_SPLITS:
        dataset_split = "tuning"
    session_label = str(form_data.get("session_label") or dataset_split).strip() or dataset_split
    auto_replay_after_capture = bool(form_data.get("auto_replay_after_capture", defaults["auto_replay_after_capture"]))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_{slugify_identifier(session_label, fallback=dataset_split)}"
    root = get_test_data_root(active_paths)
    session_dir = root / slugify_identifier(part_id) / slugify_identifier(camera_setup_id) / session_id
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "captures.jsonl"
    session_metadata_path = session_dir / "session.json"

    session_metadata = {
        "schema_version": DATASET_CAPTURE_SCHEMA_VERSION,
        "session_id": session_id,
        "project_id": project_id,
        "part_id": part_id,
        "camera_setup_id": camera_setup_id,
        "dataset_split": dataset_split,
        "session_label": session_label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inspection_mode": defaults["inspection_mode"],
        "reference_strategy": defaults["reference_strategy"],
        "config_file": str(active_paths["config_file"]),
        "reference_dir": str(active_paths["reference_dir"]),
        "config_fingerprint": compute_config_fingerprint(config),
        "roi_snapshot": defaults["roi_snapshot"],
        "capture_settings": defaults["capture_settings"],
        "hostname": socket.gethostname(),
        "auto_replay_after_capture": auto_replay_after_capture,
    }
    session_metadata_path.write_text(json.dumps(session_metadata, indent=2) + "\n", encoding="utf-8")

    return CaptureSession(
        project_id=project_id,
        part_id=part_id,
        camera_setup_id=camera_setup_id,
        dataset_split=dataset_split,
        session_label=session_label,
        session_id=session_id,
        session_dir=session_dir,
        images_dir=images_dir,
        manifest_path=manifest_path,
        session_metadata_path=session_metadata_path,
        auto_replay_after_capture=auto_replay_after_capture,
        inspection_mode=defaults["inspection_mode"],
        reference_strategy=defaults["reference_strategy"],
        config_fingerprint=compute_config_fingerprint(config),
        config_file=Path(active_paths["config_file"]),
        reference_dir=Path(active_paths["reference_dir"]),
        roi_snapshot=defaults["roi_snapshot"],
        capture_settings=defaults["capture_settings"],
    )


def append_capture_record(manifest_path: Path, record: dict) -> None:
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _count_existing_captures(images_dir: Path) -> int:
    return sum(1 for path in images_dir.iterdir() if path.is_file())


def capture_dataset_sample(
    config: dict,
    session: CaptureSession,
    *,
    bucket: str,
    defect_category: str | None = None,
    note: str | None = None,
) -> dict:
    bucket_name = bucket if bucket in CAPTURE_BUCKETS else "good"
    defect_value = str(defect_category or "").strip() or None
    if bucket_name != "good" and not defect_value:
        defect_value = "unknown"

    return_code, temp_image, capture_error = capture_to_temp(config)
    if return_code != 0 or not temp_image.exists():
        raise RuntimeError(capture_error or "Camera capture failed.")

    sequence = _count_existing_captures(session.images_dir) + 1
    suffix = f"_{slugify_identifier(defect_value)}" if defect_value else ""
    filename = f"{sequence:04d}_{slugify_identifier(bucket_name)}{suffix}.png"
    destination = session.images_dir / filename
    shutil.copy2(temp_image, destination)

    replay_result = inspect_file(config, destination) if session.auto_replay_after_capture else None
    expected_status = expected_runtime_status(bucket_name)
    actual_status = replay_result.get("status") if replay_result else None
    mismatch = bool(expected_status and actual_status and expected_status != actual_status)

    record = {
        "schema_version": DATASET_CAPTURE_SCHEMA_VERSION,
        "capture_id": f"{session.session_id}_{sequence:04d}",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sequence": sequence,
        "project_id": session.project_id,
        "part_id": session.part_id,
        "camera_setup_id": session.camera_setup_id,
        "dataset_split": session.dataset_split,
        "session_id": session.session_id,
        "session_label": session.session_label,
        "bucket": bucket_name,
        "expected_inspection_status": expected_status,
        "actual_inspection_status": actual_status,
        "result_mismatch": mismatch,
        "defect_category": defect_value,
        "note": str(note or "").strip() or None,
        "inspection_mode": session.inspection_mode,
        "reference_strategy": session.reference_strategy,
        "config_fingerprint": session.config_fingerprint,
        "config_file": str(session.config_file),
        "reference_dir": str(session.reference_dir),
        "roi_snapshot": session.roi_snapshot,
        "capture_settings": session.capture_settings,
        "hostname": socket.gethostname(),
        "image_path": str(destination),
        "relative_image_path": str(destination.relative_to(session.session_dir)),
        "replay_result": replay_result,
    }
    append_capture_record(session.manifest_path, record)
    return record


class TestDataCollectorDialog:
    """Guided image collection window launched from the dashboard."""

    def __init__(self, parent: tk.Tk, *, config_loader, active_paths_loader):
        self.parent = parent
        self.config_loader = config_loader
        self.active_paths_loader = active_paths_loader
        self.window = tk.Toplevel(parent)
        self.window.title("Collect Test Images")
        self.window.transient(parent)

        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        self.compact_layout = should_use_compact_capture_layout(self.screen_width, self.screen_height)
        self._configure_window_size()

        self.preview_image = None
        self.capture_running = False
        self.session: CaptureSession | None = None
        self.sample_inputs: list[tk.Widget] = []

        self.project_var = tk.StringVar(value="Project: -")
        self.config_var = tk.StringVar(value="Config: -")
        self.reference_var = tk.StringVar(value="Reference: -")
        self.session_var = tk.StringVar(value="Session: not started")
        self.last_result_var = tk.StringVar(value="No captures yet.")
        self.capture_guidance_var = tk.StringVar(value="Start a session to enable capture buttons and sample labeling.")
        self.part_id_var = tk.StringVar()
        self.camera_setup_id_var = tk.StringVar()
        self.session_label_var = tk.StringVar(value="commissioning")
        self.split_var = tk.StringVar(value="tuning")
        self.defect_category_var = tk.StringVar(value="")
        self.note_var = tk.StringVar(value="")
        self.auto_replay_var = tk.BooleanVar(value=True)

        self._build_layout()
        self._load_defaults()
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def _configure_window_size(self) -> None:
        width = min(980, max(720, self.screen_width - 24))
        height = min(760, max(520, self.screen_height - 24))
        self.window.geometry(f"{width}x{height}+8+8")

    def is_open(self) -> bool:
        return bool(self.window.winfo_exists())

    def focus(self) -> None:
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()

    def close(self) -> None:
        self.scroll_canvas.unbind_all("<MouseWheel>")
        self.scroll_canvas.unbind_all("<Button-4>")
        self.scroll_canvas.unbind_all("<Button-5>")
        self.window.destroy()

    def _build_layout(self) -> None:
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        shell = ttk.Frame(self.window)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        self.scroll_canvas = tk.Canvas(shell, highlightthickness=0)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self.scroll_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        self.content = ttk.Frame(self.scroll_canvas, padding=(0, 0, 0, 8))
        self.content.columnconfigure(0, weight=1)
        if not self.compact_layout:
            self.content.columnconfigure(1, weight=1)
        window_id = self.scroll_canvas.create_window((0, 0), window=self.content, anchor="nw")

        def _sync_scroll_region(_event=None) -> None:
            self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

        def _sync_content_width(event) -> None:
            self.scroll_canvas.itemconfigure(window_id, width=event.width)

        self.content.bind("<Configure>", _sync_scroll_region)
        self.scroll_canvas.bind("<Configure>", _sync_content_width)

        def _on_mousewheel(event) -> None:
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = -1 * int(event.delta / 120)
            elif getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            if delta:
                self.scroll_canvas.yview_scroll(delta, "units")

        self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.scroll_canvas.bind_all("<Button-4>", _on_mousewheel)
        self.scroll_canvas.bind_all("<Button-5>", _on_mousewheel)

        context = ttk.LabelFrame(self.content, text="Auto-filled Context", padding=10)
        context.grid(row=0, column=0, columnspan=2 if not self.compact_layout else 1, sticky="ew", padx=10, pady=(10, 6))
        context.columnconfigure(0, weight=1)
        ttk.Label(context, textvariable=self.project_var).grid(row=0, column=0, sticky="w", pady=1)
        ttk.Label(context, textvariable=self.config_var).grid(row=1, column=0, sticky="w", pady=1)
        ttk.Label(context, textvariable=self.reference_var).grid(row=2, column=0, sticky="w", pady=1)
        ttk.Label(context, textvariable=self.session_var, font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="w", pady=(6, 1))
        ttk.Label(
            context,
            textvariable=self.last_result_var,
            wraplength=460 if self.compact_layout else 760,
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(4, 1))

        session_frame = ttk.LabelFrame(self.content, text="Session Defaults", padding=10)
        session_row = 1
        session_column = 0
        session_frame.grid(row=session_row, column=session_column, sticky="nsew", padx=(10, 6), pady=(0, 6))
        session_frame.columnconfigure(1, weight=1)

        ttk.Label(session_frame, text="Part ID").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(session_frame, textvariable=self.part_id_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(session_frame, text="Camera Setup ID").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(session_frame, textvariable=self.camera_setup_id_var).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(session_frame, text="Session Label").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(session_frame, textvariable=self.session_label_var).grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Label(session_frame, text="Split").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(
            session_frame,
            textvariable=self.split_var,
            state="readonly",
            values=CAPTURE_DATASET_SPLITS,
        ).grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Checkbutton(
            session_frame,
            text="Replay immediately after each capture",
            variable=self.auto_replay_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 3))
        self.start_session_button = ttk.Button(session_frame, text="Start Session", command=self.handle_start_new_session)
        self.start_session_button.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.sample_frame = ttk.LabelFrame(self.content, text="Per-sample Metadata", padding=10)
        sample_row = 2 if self.compact_layout else 1
        sample_column = 0 if self.compact_layout else 1
        self.sample_frame.grid(row=sample_row, column=sample_column, sticky="nsew", padx=(6, 10), pady=(0, 6))
        self.sample_frame.columnconfigure(0, weight=1)
        ttk.Label(self.sample_frame, text="Defect Category").grid(row=0, column=0, sticky="w", pady=3)
        defect_combo = ttk.Combobox(
            self.sample_frame,
            textvariable=self.defect_category_var,
            state="readonly",
            values=DEFECT_CATEGORY_OPTIONS,
        )
        defect_combo.grid(row=1, column=0, sticky="ew", pady=3)
        ttk.Label(self.sample_frame, text="Optional Note").grid(row=2, column=0, sticky="w", pady=(8, 3))
        note_entry = ttk.Entry(self.sample_frame, textvariable=self.note_var)
        note_entry.grid(row=3, column=0, sticky="ew", pady=3)
        ttk.Label(
            self.sample_frame,
            text="Use defect tags like light_pipe_position or broken_coring for feature-related rejects. Leave blank for good captures.",
            wraplength=440 if self.compact_layout else 280,
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.sample_inputs.extend([defect_combo, note_entry])

        self.capture_frame = ttk.LabelFrame(self.content, text="Capture Actions", padding=10)
        capture_row = 3 if self.compact_layout else 2
        capture_column = 0
        self.capture_frame.grid(row=capture_row, column=capture_column, sticky="nsew", padx=(10, 6), pady=(0, 10))
        self.capture_frame.columnconfigure(0, weight=1)
        self.capture_frame.columnconfigure(1, weight=1)

        self.capture_buttons = []
        button_specs = [
            ("Capture Good", "good"),
            ("Capture Reject", "reject"),
            ("Capture Borderline", "borderline"),
            ("Capture Invalid", "invalid_capture"),
        ]
        for index, (label, bucket) in enumerate(button_specs):
            button = ttk.Button(self.capture_frame, text=label, command=lambda value=bucket: self.begin_capture(value))
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=4, pady=6)
            self.capture_buttons.append(button)

        ttk.Label(
            self.capture_frame,
            textvariable=self.capture_guidance_var,
            wraplength=440 if self.compact_layout else 520,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.preview_frame = ttk.LabelFrame(self.content, text="Latest Capture", padding=10)
        preview_row = 4 if self.compact_layout else 2
        preview_column = 0 if self.compact_layout else 1
        self.preview_frame.grid(row=preview_row, column=preview_column, sticky="nsew", padx=(6, 10), pady=(0, 10))
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(self.preview_frame, text="No capture yet.", anchor="center", justify="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        self.recent_frame = ttk.LabelFrame(self.content, text="Recent Records", padding=10)
        recent_row = 5 if self.compact_layout else 3
        recent_span = 1 if self.compact_layout else 2
        self.recent_frame.grid(row=recent_row, column=0, columnspan=recent_span, sticky="nsew", padx=10, pady=(0, 10))
        self.recent_frame.columnconfigure(0, weight=1)
        self.recent_frame.rowconfigure(0, weight=1)
        self.recent_list = tk.Listbox(self.recent_frame, height=4 if self.compact_layout else 8)
        self.recent_list.grid(row=0, column=0, sticky="nsew")

        self._set_collection_controls_enabled(False)

    def _load_defaults(self) -> None:
        config = self.config_loader()
        active_paths = self.active_paths_loader()
        project_name = get_current_project() or "default"
        defaults = build_collection_defaults(config, active_paths, project_name)
        self.project_var.set(f"Project: {project_name}")
        self.config_var.set(f"Config: {format_display_path(active_paths['config_file'], keep_parts=3)}")
        self.reference_var.set(f"Reference: {format_display_path(active_paths['reference_dir'], keep_parts=2)}")
        self.part_id_var.set(defaults["part_id"])
        self.camera_setup_id_var.set(defaults["camera_setup_id"])
        self.split_var.set(defaults["default_split"])
        self.auto_replay_var.set(defaults["auto_replay_after_capture"])

    def _set_capture_state(self, busy: bool) -> None:
        self.capture_running = busy
        state = "disabled" if busy else "normal"
        for button in self.capture_buttons:
            button.configure(state=state)

    def _set_sample_inputs_state(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        self.sample_inputs[0].configure(state=state)
        self.sample_inputs[1].configure(state="normal" if enabled else "disabled")

    def _set_collection_controls_enabled(self, enabled: bool) -> None:
        self._set_sample_inputs_state(enabled)
        self._set_capture_state(not enabled)
        if enabled:
            self.capture_guidance_var.set(
                "Capture buttons are active. Defect category and note apply to the next captured sample."
            )
        else:
            self.capture_guidance_var.set(
                "Start a session to enable capture buttons and sample labeling."
            )

    def handle_start_new_session(self) -> None:
        try:
            self.start_new_session()
        except Exception as exc:
            self.last_result_var.set(f"Session start failed: {exc}")
            messagebox.showerror("Session error", f"Could not start a capture session: {exc}")

    def start_new_session(self) -> None:
        config = self.config_loader()
        active_paths = self.active_paths_loader()
        config_file = Path(active_paths["config_file"])
        persist_collection_defaults(
            config_file,
            config,
            part_id=self.part_id_var.get(),
            camera_setup_id=self.camera_setup_id_var.get(),
            default_split=self.split_var.get(),
            auto_replay_after_capture=self.auto_replay_var.get(),
        )
        refreshed_config = self.config_loader()
        self.session = create_capture_session(
            refreshed_config,
            active_paths,
            get_current_project(),
            {
                "part_id": self.part_id_var.get(),
                "camera_setup_id": self.camera_setup_id_var.get(),
                "dataset_split": self.split_var.get(),
                "session_label": self.session_label_var.get(),
                "auto_replay_after_capture": self.auto_replay_var.get(),
            },
        )
        self.session_var.set(f"Session: {self.session.session_id} | Split: {self.session.dataset_split}")
        self.last_result_var.set(
            f"Session started. Saving to {format_display_path(self.session.session_dir, keep_parts=4)}"
        )
        self.start_session_button.configure(text="Start New Session")
        self._set_collection_controls_enabled(True)
        self.defect_category_var.set("")
        self.note_var.set("")
        self.sample_inputs[0].focus_set()

    def _ensure_session(self) -> bool:
        if self.session is not None:
            return True
        try:
            self.start_new_session()
            return True
        except Exception as exc:
            messagebox.showerror("Session error", f"Could not start a capture session: {exc}")
            return False

    def begin_capture(self, bucket: str) -> None:
        if self.capture_running:
            return
        if not self._ensure_session() or self.session is None:
            return

        defect_category = self.defect_category_var.get().strip() or None
        note = self.note_var.get().strip() or None
        session = self.session
        config = self.config_loader()
        self._set_capture_state(True)
        self.last_result_var.set(f"Capturing {bucket} sample...")
        threading.Thread(
            target=self._capture_thread,
            args=(config, session, bucket, defect_category, note),
            daemon=True,
        ).start()

    def _capture_thread(self, config: dict, session: CaptureSession, bucket: str, defect_category: str | None, note: str | None) -> None:
        try:
            record = capture_dataset_sample(
                config,
                session,
                bucket=bucket,
                defect_category=defect_category,
                note=note,
            )
        except Exception as exc:
            self.window.after(0, self._capture_failed, str(exc))
            return
        self.window.after(0, self._capture_finished, record)

    def _capture_failed(self, error_text: str) -> None:
        self._set_capture_state(False)
        self.last_result_var.set(f"Capture failed: {error_text}")
        messagebox.showerror("Capture failed", error_text)

    def _capture_finished(self, record: dict) -> None:
        self._set_capture_state(False)
        expected = record.get("expected_inspection_status") or "n/a"
        actual = record.get("actual_inspection_status") or "not replayed"
        mismatch = " | mismatch" if record.get("result_mismatch") else ""
        self.last_result_var.set(
            f"Saved {Path(record['image_path']).name} | bucket={record['bucket']} | expected={expected} | actual={actual}{mismatch}"
        )
        self.recent_list.insert(
            0,
            f"{Path(record['image_path']).name} | {record['bucket']} | defect={record.get('defect_category') or '-'} | actual={actual}",
        )
        while self.recent_list.size() > 12:
            self.recent_list.delete(tk.END)
        self.note_var.set("")
        self._update_preview(Path(record["image_path"]))

    def _update_preview(self, image_path: Path) -> None:
        if not PIL_CAPTURE_PREVIEW:
            self.preview_label.configure(text=str(image_path), image="")
            return
        try:
            image = Image.open(image_path)
            image.thumbnail((360, 280))
            self.preview_image = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_image, text="")
        except Exception:
            self.preview_label.configure(text=str(image_path), image="")
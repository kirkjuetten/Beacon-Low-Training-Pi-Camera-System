from __future__ import annotations

from dataclasses import dataclass, replace
import math

from inspection_system.app.alignment_utils import get_mask_centroid_and_angle, rotate_mask, shift_mask
from inspection_system.app.registration_schema import get_registration_config
from inspection_system.app.registration_transform import apply_transform_to_point, build_transform_summary


@dataclass(frozen=True)
class RegistrationResult:
    aligned_mask: object
    angle_deg: float
    shift_x: int
    shift_y: int
    enabled: bool
    status: str
    runtime_mode: str
    requested_strategy: str
    applied_strategy: str
    transform_model: str
    anchor_mode: str
    subpixel_refinement: str
    fallback_reason: str | None
    rejection_reason: str | None
    quality: dict
    quality_gates: dict
    quality_gate_failures: list[dict]
    datum_frame: dict
    transform: dict
    observed_anchors: list[dict]


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _build_registration_quality_gate_failures(quality: dict, quality_gates: dict) -> list[dict]:
    failures: list[dict] = []

    confidence = _optional_finite_float(quality.get("confidence"))
    min_confidence = _optional_finite_float(quality_gates.get("min_confidence"))
    if min_confidence is not None and (confidence is None or confidence < min_confidence):
        failures.append(
            {
                "cause_code": "registration_failure",
                "gate_key": "min_confidence",
                "title": "registration confidence",
                "metric_key": "confidence",
                "threshold": min_confidence,
                "observed": confidence,
                "margin": None if confidence is None else float(confidence - min_confidence),
                "summary": (
                    f"Registration confidence {confidence:.3f} was below required minimum {min_confidence:.3f}."
                    if confidence is not None
                    else f"Registration confidence was unavailable, but minimum {min_confidence:.3f} is required."
                ),
            }
        )

    mean_residual_px = _optional_finite_float(quality.get("mean_residual_px"))
    max_mean_residual_px = _optional_finite_float(quality_gates.get("max_mean_residual_px"))
    if max_mean_residual_px is not None and (mean_residual_px is None or mean_residual_px > max_mean_residual_px):
        failures.append(
            {
                "cause_code": "registration_failure",
                "gate_key": "max_mean_residual_px",
                "title": "registration mean residual",
                "metric_key": "mean_residual_px",
                "threshold": max_mean_residual_px,
                "observed": mean_residual_px,
                "margin": None if mean_residual_px is None else float(max_mean_residual_px - mean_residual_px),
                "summary": (
                    f"Registration mean residual {mean_residual_px:.3f}px exceeded allowed maximum {max_mean_residual_px:.3f}px."
                    if mean_residual_px is not None
                    else f"Registration mean residual was unavailable, but maximum {max_mean_residual_px:.3f}px is required."
                ),
            }
        )

    return failures


def _finalize_registration_result(result: RegistrationResult) -> RegistrationResult:
    if result.status != "aligned":
        return result

    quality_gate_failures = _build_registration_quality_gate_failures(result.quality, result.quality_gates)
    if not quality_gate_failures:
        return replace(result, rejection_reason=None, quality_gate_failures=[])

    return replace(
        result,
        status="quality_gate_failed",
        rejection_reason="; ".join(failure["summary"] for failure in quality_gate_failures),
        quality_gate_failures=quality_gate_failures,
    )


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _resolve_search_window(mask_shape: tuple[int, int], anchor: dict, search_margin_px: int) -> tuple[int, int, int, int]:
    height, width = mask_shape[:2]
    reference_point = anchor.get("reference_point", {})
    anchor_x = int(reference_point.get("x", 0))
    anchor_y = int(reference_point.get("y", 0))
    search_window = anchor.get("search_window", {}) if isinstance(anchor.get("search_window"), dict) else {}

    window_x = int(search_window.get("x", anchor_x - search_margin_px))
    window_y = int(search_window.get("y", anchor_y - search_margin_px))
    window_width = int(search_window.get("width", 0) or (search_margin_px * 2 + 1))
    window_height = int(search_window.get("height", 0) or (search_margin_px * 2 + 1))

    x0 = max(0, min(width, window_x))
    y0 = max(0, min(height, window_y))
    x1 = max(x0, min(width, x0 + max(1, window_width)))
    y1 = max(y0, min(height, y0 + max(1, window_height)))
    return x0, y0, x1, y1


def _locate_sample_anchor(mask, anchor: dict, search_margin_px: int, np) -> tuple[tuple[float, float] | None, float]:
    x0, y0, x1, y1 = _resolve_search_window(mask.shape[:2], anchor, search_margin_px)
    window = mask[y0:y1, x0:x1] > 0
    points = np.argwhere(window)
    if points.size == 0:
        return None, 0.0

    centroid_y = float(points[:, 0].mean() + y0)
    centroid_x = float(points[:, 1].mean() + x0)
    return (centroid_x, centroid_y), 1.0


def _build_failed_registration_result(
    sample_mask,
    *,
    enabled: bool,
    runtime_mode: str,
    requested_strategy: str,
    transform_model: str,
    anchor_mode: str,
    subpixel_refinement: str,
    fallback_reason: str,
    rejection_reason: str | None,
    quality: dict,
    quality_gates: dict,
    datum_frame: dict,
    status: str,
    observed_anchors: list[dict] | None = None,
) -> RegistrationResult:
    return RegistrationResult(
        aligned_mask=sample_mask,
        angle_deg=0.0,
        shift_x=0,
        shift_y=0,
        enabled=enabled,
        status=status,
        runtime_mode=runtime_mode,
        requested_strategy=requested_strategy,
        applied_strategy="identity",
        transform_model=transform_model,
        anchor_mode=anchor_mode,
        subpixel_refinement=subpixel_refinement,
        fallback_reason=fallback_reason,
        rejection_reason=rejection_reason,
        quality=quality,
        quality_gates=quality_gates,
        quality_gate_failures=[],
        datum_frame=datum_frame,
        transform=build_transform_summary(sample_mask.shape[:2], 0.0, 0, 0) if hasattr(sample_mask, "shape") else build_transform_summary((1, 1), 0.0, 0, 0),
        observed_anchors=list(observed_anchors or []),
    )


def _apply_rigid_transform_to_mask(mask, angle_deg: float, shift_x: int, shift_y: int, cv2, np):
    rotated_mask = rotate_mask(mask, angle_deg, cv2) if abs(angle_deg) > 1e-6 else mask
    if shift_x or shift_y:
        return shift_mask(rotated_mask, shift_x, shift_y, cv2, np)
    return rotated_mask


def _compute_mask_overlap_confidence(reference_mask, sample_mask, np) -> float:
    reference_white = reference_mask > 0
    sample_white = sample_mask > 0
    reference_count = int(reference_white.sum())
    sample_count = int(sample_white.sum())
    if reference_count == 0 and sample_count == 0:
        return 1.0
    if reference_count == 0 or sample_count == 0:
        return 0.0
    overlap = int((reference_white & sample_white).sum())
    return float((2.0 * overlap) / float(reference_count + sample_count))


def _compute_mask_centroid_residual(reference_mask, sample_mask, cv2, np) -> float | None:
    if not hasattr(cv2, "findNonZero") or not hasattr(cv2, "moments"):
        return None
    reference_centroid, _ = get_mask_centroid_and_angle(reference_mask, cv2, np)
    sample_centroid, _ = get_mask_centroid_and_angle(sample_mask, cv2, np)
    if reference_centroid is None or sample_centroid is None:
        return None
    return float(math.hypot(reference_centroid[0] - sample_centroid[0], reference_centroid[1] - sample_centroid[1]))


def _build_rigid_refinement_candidates(
    coarse_angle: float,
    coarse_shift_x: int,
    coarse_shift_y: int,
    alignment_cfg: dict,
    registration_cfg: dict,
) -> list[tuple[float, int, int]]:
    max_angle_deg = float(alignment_cfg.get("max_angle_deg", 1.0))
    max_shift_x = int(alignment_cfg.get("max_shift_x", 4))
    max_shift_y = int(alignment_cfg.get("max_shift_y", 3))
    refinement_mode = str(registration_cfg.get("subpixel_refinement", "off")).lower()

    angle_offsets = [0.0, -0.25, 0.25]
    shift_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    if refinement_mode in {"template", "phase_correlation"}:
        angle_offsets = [0.0, -0.5, -0.25, 0.25, 0.5]
        shift_offsets = [
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
        ]

    candidates: list[tuple[float, int, int]] = []
    seen: set[tuple[float, int, int]] = set()
    for angle_offset in angle_offsets:
        angle = round(_clamp(coarse_angle + angle_offset, max_angle_deg), 6)
        for shift_offset_x, shift_offset_y in shift_offsets:
            shift_x = int(round(_clamp(coarse_shift_x + shift_offset_x, max_shift_x)))
            shift_y = int(round(_clamp(coarse_shift_y + shift_offset_y, max_shift_y)))
            candidate = (angle, shift_x, shift_y)
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _register_with_rigid_refined(sample_mask, reference_mask, alignment_cfg: dict, registration_cfg: dict, cv2, np, align_sample_mask_fn) -> RegistrationResult:
    coarse_alignment_cfg = dict(alignment_cfg)
    coarse_alignment_cfg["mode"] = "moments"
    coarse_aligned_mask, coarse_angle_deg, coarse_shift_x, coarse_shift_y = align_sample_mask_fn(
        sample_mask,
        reference_mask,
        coarse_alignment_cfg,
        cv2,
        np,
    )

    best_angle_deg = float(coarse_angle_deg)
    best_shift_x = int(coarse_shift_x)
    best_shift_y = int(coarse_shift_y)
    best_aligned_mask = coarse_aligned_mask
    best_confidence = _compute_mask_overlap_confidence(reference_mask, coarse_aligned_mask, np)
    best_residual = _compute_mask_centroid_residual(reference_mask, coarse_aligned_mask, cv2, np)

    for candidate_angle_deg, candidate_shift_x, candidate_shift_y in _build_rigid_refinement_candidates(
        best_angle_deg,
        best_shift_x,
        best_shift_y,
        alignment_cfg,
        registration_cfg,
    ):
        if (
            abs(candidate_angle_deg - best_angle_deg) < 1e-6
            and candidate_shift_x == best_shift_x
            and candidate_shift_y == best_shift_y
        ):
            continue
        candidate_mask = _apply_rigid_transform_to_mask(
            sample_mask,
            candidate_angle_deg,
            candidate_shift_x,
            candidate_shift_y,
            cv2,
            np,
        )
        candidate_confidence = _compute_mask_overlap_confidence(reference_mask, candidate_mask, np)
        candidate_residual = _compute_mask_centroid_residual(reference_mask, candidate_mask, cv2, np)
        candidate_residual_rank = float("inf") if candidate_residual is None else candidate_residual
        best_residual_rank = float("inf") if best_residual is None else best_residual
        if (
            candidate_confidence > best_confidence + 1e-6
            or (
                abs(candidate_confidence - best_confidence) <= 1e-6
                and candidate_residual_rank < best_residual_rank
            )
        ):
            best_angle_deg = float(candidate_angle_deg)
            best_shift_x = int(candidate_shift_x)
            best_shift_y = int(candidate_shift_y)
            best_aligned_mask = candidate_mask
            best_confidence = float(candidate_confidence)
            best_residual = candidate_residual

    return _finalize_registration_result(RegistrationResult(
        aligned_mask=best_aligned_mask,
        angle_deg=best_angle_deg,
        shift_x=best_shift_x,
        shift_y=best_shift_y,
        enabled=True,
        status="aligned",
        runtime_mode="rigid_refined",
        requested_strategy=registration_cfg["strategy"],
        applied_strategy="rigid_refined",
        transform_model=registration_cfg["transform_model"],
        anchor_mode=registration_cfg["anchor_mode"],
        subpixel_refinement=registration_cfg["subpixel_refinement"],
        fallback_reason=None,
        rejection_reason=None,
        quality={
            "confidence": float(best_confidence),
            "mean_residual_px": None if best_residual is None else float(best_residual),
        },
        quality_gates=dict(registration_cfg["quality_gates"]),
        quality_gate_failures=[],
        datum_frame=dict(registration_cfg["datum_frame"]),
        transform=build_transform_summary(sample_mask.shape[:2], best_angle_deg, best_shift_x, best_shift_y),
        observed_anchors=[],
    ))


def _register_with_anchor_translation(sample_mask, reference_mask, alignment_cfg: dict, registration_cfg: dict, cv2, np) -> RegistrationResult:
    anchors = [anchor for anchor in registration_cfg["anchors"] if anchor.get("enabled", True)]
    if not anchors:
        return _build_failed_registration_result(
            sample_mask,
            enabled=True,
            runtime_mode="anchor_translation",
            requested_strategy=registration_cfg["strategy"],
            transform_model=registration_cfg["transform_model"],
            anchor_mode=registration_cfg["anchor_mode"],
            subpixel_refinement=registration_cfg["subpixel_refinement"],
            fallback_reason="Anchor translation requires at least one enabled anchor.",
            rejection_reason="Anchor translation requires at least one enabled anchor.",
            quality={"confidence": 0.0, "mean_residual_px": None},
            quality_gates=dict(registration_cfg["quality_gates"]),
            datum_frame=dict(registration_cfg["datum_frame"]),
            status="anchor_configuration_missing",
        )

    anchor = anchors[0]
    sample_anchor, confidence = _locate_sample_anchor(sample_mask, anchor, registration_cfg["search_margin_px"], np)
    if sample_anchor is None:
        return _build_failed_registration_result(
            sample_mask,
            enabled=True,
            runtime_mode="anchor_translation",
            requested_strategy=registration_cfg["strategy"],
            transform_model=registration_cfg["transform_model"],
            anchor_mode=registration_cfg["anchor_mode"],
            subpixel_refinement=registration_cfg["subpixel_refinement"],
            fallback_reason=f"Could not localize sample anchor '{anchor['anchor_id']}' in its search window.",
            rejection_reason=f"Could not localize sample anchor '{anchor['anchor_id']}' in its search window.",
            quality={"confidence": 0.0, "mean_residual_px": None},
            quality_gates=dict(registration_cfg["quality_gates"]),
            datum_frame=dict(registration_cfg["datum_frame"]),
            status="anchor_localization_failed",
            observed_anchors=[{"anchor_id": anchor["anchor_id"], "reference_point": dict(anchor["reference_point"]), "sample_point": None}],
        )

    reference_point = anchor["reference_point"]
    shift_x = int(round(reference_point["x"] - sample_anchor[0]))
    shift_y = int(round(reference_point["y"] - sample_anchor[1]))
    max_shift_x = int(alignment_cfg.get("max_shift_x", 4))
    max_shift_y = int(alignment_cfg.get("max_shift_y", 3))
    shift_x = int(_clamp(shift_x, max_shift_x))
    shift_y = int(_clamp(shift_y, max_shift_y))
    aligned_mask = shift_mask(sample_mask, shift_x, shift_y, cv2, np) if (shift_x or shift_y) else sample_mask
    residual = math.hypot(reference_point["x"] - (sample_anchor[0] + shift_x), reference_point["y"] - (sample_anchor[1] + shift_y))
    transformed_anchor = apply_transform_to_point(sample_anchor, sample_mask.shape[:2], 0.0, shift_x, shift_y)
    return _finalize_registration_result(RegistrationResult(
        aligned_mask=aligned_mask,
        angle_deg=0.0,
        shift_x=shift_x,
        shift_y=shift_y,
        enabled=True,
        status="aligned",
        runtime_mode="anchor_translation",
        requested_strategy=registration_cfg["strategy"],
        applied_strategy="anchor_translation",
        transform_model=registration_cfg["transform_model"],
        anchor_mode=registration_cfg["anchor_mode"],
        subpixel_refinement=registration_cfg["subpixel_refinement"],
        fallback_reason=None,
        rejection_reason=None,
        quality={"confidence": float(confidence), "mean_residual_px": float(residual)},
        quality_gates=dict(registration_cfg["quality_gates"]),
        quality_gate_failures=[],
        datum_frame=dict(registration_cfg["datum_frame"]),
        transform=build_transform_summary(sample_mask.shape[:2], 0.0, shift_x, shift_y),
        observed_anchors=[
            {
                "anchor_id": anchor["anchor_id"],
                "reference_point": dict(anchor["reference_point"]),
                "sample_point": {"x": float(sample_anchor[0]), "y": float(sample_anchor[1])},
                "transformed_point": {"x": float(transformed_anchor[0]), "y": float(transformed_anchor[1])},
            }
        ],
    ))


def _register_with_anchor_pair(sample_mask, reference_mask, alignment_cfg: dict, registration_cfg: dict, cv2, np) -> RegistrationResult:
    anchors = [anchor for anchor in registration_cfg["anchors"] if anchor.get("enabled", True)]
    if len(anchors) < 2:
        return _build_failed_registration_result(
            sample_mask,
            enabled=True,
            runtime_mode="anchor_pair",
            requested_strategy=registration_cfg["strategy"],
            transform_model=registration_cfg["transform_model"],
            anchor_mode=registration_cfg["anchor_mode"],
            subpixel_refinement=registration_cfg["subpixel_refinement"],
            fallback_reason="Anchor pair registration requires at least two enabled anchors.",
            rejection_reason="Anchor pair registration requires at least two enabled anchors.",
            quality={"confidence": 0.0, "mean_residual_px": None},
            quality_gates=dict(registration_cfg["quality_gates"]),
            datum_frame=dict(registration_cfg["datum_frame"]),
            status="anchor_configuration_missing",
        )

    primary_anchor, secondary_anchor = anchors[:2]
    primary_sample, primary_confidence = _locate_sample_anchor(sample_mask, primary_anchor, registration_cfg["search_margin_px"], np)
    secondary_sample, secondary_confidence = _locate_sample_anchor(sample_mask, secondary_anchor, registration_cfg["search_margin_px"], np)
    if primary_sample is None or secondary_sample is None:
        missing_anchor = primary_anchor["anchor_id"] if primary_sample is None else secondary_anchor["anchor_id"]
        return _build_failed_registration_result(
            sample_mask,
            enabled=True,
            runtime_mode="anchor_pair",
            requested_strategy=registration_cfg["strategy"],
            transform_model=registration_cfg["transform_model"],
            anchor_mode=registration_cfg["anchor_mode"],
            subpixel_refinement=registration_cfg["subpixel_refinement"],
            fallback_reason=f"Could not localize sample anchor '{missing_anchor}' in its search window.",
            rejection_reason=f"Could not localize sample anchor '{missing_anchor}' in its search window.",
            quality={"confidence": 0.0, "mean_residual_px": None},
            quality_gates=dict(registration_cfg["quality_gates"]),
            datum_frame=dict(registration_cfg["datum_frame"]),
            status="anchor_localization_failed",
            observed_anchors=[
                {
                    "anchor_id": primary_anchor["anchor_id"],
                    "reference_point": dict(primary_anchor["reference_point"]),
                    "sample_point": None if primary_sample is None else {"x": float(primary_sample[0]), "y": float(primary_sample[1])},
                },
                {
                    "anchor_id": secondary_anchor["anchor_id"],
                    "reference_point": dict(secondary_anchor["reference_point"]),
                    "sample_point": None if secondary_sample is None else {"x": float(secondary_sample[0]), "y": float(secondary_sample[1])},
                },
            ],
        )

    primary_ref = (float(primary_anchor["reference_point"]["x"]), float(primary_anchor["reference_point"]["y"]))
    secondary_ref = (float(secondary_anchor["reference_point"]["x"]), float(secondary_anchor["reference_point"]["y"]))

    sample_vector = (secondary_sample[0] - primary_sample[0], secondary_sample[1] - primary_sample[1])
    reference_vector = (secondary_ref[0] - primary_ref[0], secondary_ref[1] - primary_ref[1])
    if math.hypot(*sample_vector) < 1e-6 or math.hypot(*reference_vector) < 1e-6:
        return _build_failed_registration_result(
            sample_mask,
            enabled=True,
            runtime_mode="anchor_pair",
            requested_strategy=registration_cfg["strategy"],
            transform_model=registration_cfg["transform_model"],
            anchor_mode=registration_cfg["anchor_mode"],
            subpixel_refinement=registration_cfg["subpixel_refinement"],
            fallback_reason="Anchor pair registration requires separated reference and sample anchors.",
            rejection_reason="Anchor pair registration requires separated reference and sample anchors.",
            quality={"confidence": 0.0, "mean_residual_px": None},
            quality_gates=dict(registration_cfg["quality_gates"]),
            datum_frame=dict(registration_cfg["datum_frame"]),
            status="anchor_geometry_invalid",
        )

    sample_angle_deg = math.degrees(math.atan2(sample_vector[1], sample_vector[0]))
    reference_angle_deg = math.degrees(math.atan2(reference_vector[1], reference_vector[0]))
    max_angle_deg = float(alignment_cfg.get("max_angle_deg", 1.0))
    angle_delta = _clamp(reference_angle_deg - sample_angle_deg, max_angle_deg)
    rotated_mask = rotate_mask(sample_mask, angle_delta, cv2) if abs(angle_delta) > 1e-6 else sample_mask

    rotated_primary = apply_transform_to_point(primary_sample, sample_mask.shape[:2], angle_delta, 0, 0)
    rotated_secondary = apply_transform_to_point(secondary_sample, sample_mask.shape[:2], angle_delta, 0, 0)
    shift_x_float = ((primary_ref[0] - rotated_primary[0]) + (secondary_ref[0] - rotated_secondary[0])) / 2.0
    shift_y_float = ((primary_ref[1] - rotated_primary[1]) + (secondary_ref[1] - rotated_secondary[1])) / 2.0

    max_shift_x = int(alignment_cfg.get("max_shift_x", 4))
    max_shift_y = int(alignment_cfg.get("max_shift_y", 3))
    shift_x = int(round(_clamp(shift_x_float, max_shift_x)))
    shift_y = int(round(_clamp(shift_y_float, max_shift_y)))
    aligned_mask = shift_mask(rotated_mask, shift_x, shift_y, cv2, np) if (shift_x or shift_y) else rotated_mask

    transformed_primary = (rotated_primary[0] + shift_x, rotated_primary[1] + shift_y)
    transformed_secondary = (rotated_secondary[0] + shift_x, rotated_secondary[1] + shift_y)
    residual_primary = math.hypot(primary_ref[0] - transformed_primary[0], primary_ref[1] - transformed_primary[1])
    residual_secondary = math.hypot(secondary_ref[0] - transformed_secondary[0], secondary_ref[1] - transformed_secondary[1])
    residual = (residual_primary + residual_secondary) / 2.0
    confidence = (primary_confidence + secondary_confidence) / 2.0
    transformed_primary = apply_transform_to_point(primary_sample, sample_mask.shape[:2], angle_delta, shift_x, shift_y)
    transformed_secondary = apply_transform_to_point(secondary_sample, sample_mask.shape[:2], angle_delta, shift_x, shift_y)

    return _finalize_registration_result(RegistrationResult(
        aligned_mask=aligned_mask,
        angle_deg=float(angle_delta),
        shift_x=shift_x,
        shift_y=shift_y,
        enabled=True,
        status="aligned",
        runtime_mode="anchor_pair",
        requested_strategy=registration_cfg["strategy"],
        applied_strategy="anchor_pair",
        transform_model=registration_cfg["transform_model"],
        anchor_mode=registration_cfg["anchor_mode"],
        subpixel_refinement=registration_cfg["subpixel_refinement"],
        fallback_reason=None,
        rejection_reason=None,
        quality={"confidence": float(confidence), "mean_residual_px": float(residual)},
        quality_gates=dict(registration_cfg["quality_gates"]),
        quality_gate_failures=[],
        datum_frame=dict(registration_cfg["datum_frame"]),
        transform=build_transform_summary(sample_mask.shape[:2], angle_delta, shift_x, shift_y),
        observed_anchors=[
            {
                "anchor_id": primary_anchor["anchor_id"],
                "reference_point": dict(primary_anchor["reference_point"]),
                "sample_point": {"x": float(primary_sample[0]), "y": float(primary_sample[1])},
                "transformed_point": {"x": float(transformed_primary[0]), "y": float(transformed_primary[1])},
            },
            {
                "anchor_id": secondary_anchor["anchor_id"],
                "reference_point": dict(secondary_anchor["reference_point"]),
                "sample_point": {"x": float(secondary_sample[0]), "y": float(secondary_sample[1])},
                "transformed_point": {"x": float(transformed_secondary[0]), "y": float(transformed_secondary[1])},
            },
        ],
    ))


def register_sample_mask(sample_mask, reference_mask, alignment_cfg: dict, cv2, np, align_sample_mask_fn) -> RegistrationResult:
    registration_cfg = get_registration_config({"alignment": alignment_cfg})
    enabled = bool(alignment_cfg.get("enabled", True))
    runtime_mode = str(alignment_cfg.get("mode", "moments")).lower()
    requested_strategy = registration_cfg["strategy"]
    fallback_reason = None
    status = "aligned"
    applied_strategy = runtime_mode if enabled else "identity"
    aligned_mask = sample_mask
    angle_deg = 0.0
    shift_x = 0
    shift_y = 0
    quality = {
        "confidence": None,
        "mean_residual_px": None,
    }

    if not enabled:
        status = "disabled"
        applied_strategy = "identity"
    elif runtime_mode == "anchor_translation":
        return _register_with_anchor_translation(sample_mask, reference_mask, alignment_cfg, registration_cfg, cv2, np)
    elif runtime_mode == "anchor_pair":
        return _register_with_anchor_pair(sample_mask, reference_mask, alignment_cfg, registration_cfg, cv2, np)
    elif runtime_mode == "rigid_refined":
        return _register_with_rigid_refined(
            sample_mask,
            reference_mask,
            alignment_cfg,
            registration_cfg,
            cv2,
            np,
            align_sample_mask_fn,
        )
    elif runtime_mode != "moments":
        status = "pending_runtime_support"
        applied_strategy = "identity"
        fallback_reason = f"Runtime alignment mode '{runtime_mode}' is not implemented; using identity registration."
    else:
        aligned_mask, angle_deg, shift_x, shift_y = align_sample_mask_fn(
            sample_mask,
            reference_mask,
            alignment_cfg,
            cv2,
            np,
        )
        if requested_strategy != runtime_mode:
            fallback_reason = (
                f"Requested registration strategy '{requested_strategy}' is staged but runtime is using '{runtime_mode}'."
            )
        quality = {
            "confidence": _compute_mask_overlap_confidence(reference_mask, aligned_mask, np),
            "mean_residual_px": _compute_mask_centroid_residual(reference_mask, aligned_mask, cv2, np),
        }

    return _finalize_registration_result(RegistrationResult(
        aligned_mask=aligned_mask,
        angle_deg=float(angle_deg),
        shift_x=int(shift_x),
        shift_y=int(shift_y),
        enabled=enabled,
        status=status,
        runtime_mode=runtime_mode,
        requested_strategy=requested_strategy,
        applied_strategy=applied_strategy,
        transform_model=registration_cfg["transform_model"],
        anchor_mode=registration_cfg["anchor_mode"],
        subpixel_refinement=registration_cfg["subpixel_refinement"],
        fallback_reason=fallback_reason,
        rejection_reason=None,
        quality=quality,
        quality_gates=dict(registration_cfg["quality_gates"]),
        quality_gate_failures=[],
        datum_frame=dict(registration_cfg["datum_frame"]),
        transform=build_transform_summary(sample_mask.shape[:2], angle_deg, shift_x, shift_y) if hasattr(sample_mask, "shape") else build_transform_summary((1, 1), angle_deg, shift_x, shift_y),
        observed_anchors=[],
    ))
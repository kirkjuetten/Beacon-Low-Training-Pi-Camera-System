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


def _resolve_anchor_template_bounds(mask_shape: tuple[int, int], anchor: dict, search_margin_px: int) -> tuple[int, int, int, int]:
    height, width = mask_shape[:2]
    ref_point = anchor.get("reference_point", {})
    anchor_x = int(ref_point.get("x", 0))
    anchor_y = int(ref_point.get("y", 0))
    search_x0, search_y0, search_x1, search_y1 = _resolve_search_window(mask_shape, anchor, search_margin_px)
    search_width = max(1, search_x1 - search_x0)
    search_height = max(1, search_y1 - search_y0)

    template_width = search_width if search_width <= 5 else max(5, int(round(search_width * 0.6)))
    template_height = search_height if search_height <= 5 else max(5, int(round(search_height * 0.6)))
    template_width = min(search_width, template_width)
    template_height = min(search_height, template_height)

    x0 = max(0, min(width - template_width, anchor_x - (template_width // 2)))
    y0 = max(0, min(height - template_height, anchor_y - (template_height // 2)))
    x1 = min(width, x0 + template_width)
    y1 = min(height, y0 + template_height)
    return x0, y0, x1, y1


def _weighted_correlation_score(template_image, candidate_image, weights, np) -> float:
    template = template_image.astype(np.float32, copy=False)
    candidate = candidate_image.astype(np.float32, copy=False)
    weights = weights.astype(np.float32, copy=False)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-6:
        weights = np.ones_like(template, dtype=np.float32)
        weight_sum = float(weights.sum())

    template_mean = float((template * weights).sum() / weight_sum)
    candidate_mean = float((candidate * weights).sum() / weight_sum)
    template_centered = (template - template_mean) * weights
    candidate_centered = (candidate - candidate_mean) * weights
    template_energy = float((template_centered * template_centered).sum())
    candidate_energy = float((candidate_centered * candidate_centered).sum())

    if template_energy <= 1e-6 or candidate_energy <= 1e-6:
        mean_abs_error = float((np.abs(template - candidate) * weights).sum() / weight_sum)
        return max(0.0, min(1.0, 1.0 - (mean_abs_error / 255.0)))

    correlation = float((template_centered * candidate_centered).sum() / math.sqrt(template_energy * candidate_energy))
    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


def _dice_overlap_score(reference_mask_patch, sample_mask_patch, np) -> float:
    reference_white = reference_mask_patch > 0
    sample_white = sample_mask_patch > 0
    reference_count = int(reference_white.sum())
    sample_count = int(sample_white.sum())
    if reference_count == 0 and sample_count == 0:
        return 1.0
    if reference_count == 0 or sample_count == 0:
        return 0.0
    overlap = int((reference_white & sample_white).sum())
    return float((2.0 * overlap) / float(reference_count + sample_count))


def _match_anchor_template(
    sample_mask,
    reference_mask,
    sample_registration_image,
    reference_registration_image,
    anchor: dict,
    search_margin_px: int,
    np,
) -> dict | None:
    if sample_registration_image is None or reference_registration_image is None:
        return None

    x0, y0, x1, y1 = _resolve_search_window(sample_mask.shape[:2], anchor, search_margin_px)
    tx0, ty0, tx1, ty1 = _resolve_anchor_template_bounds(reference_mask.shape[:2], anchor, search_margin_px)
    if tx1 <= tx0 or ty1 <= ty0:
        return None

    template_image = reference_registration_image[ty0:ty1, tx0:tx1]
    template_mask = reference_mask[ty0:ty1, tx0:tx1]
    if template_image.size == 0:
        return None

    template_feature_pixels = int((template_mask > 0).sum())
    template_signal = float(template_image.max()) - float(template_image.min())
    if template_signal <= 1e-6 and template_feature_pixels == 0:
        return None
    if template_feature_pixels < 5:
        return None

    search_image = sample_registration_image[y0:y1, x0:x1]
    search_mask = sample_mask[y0:y1, x0:x1]
    template_height, template_width = template_image.shape[:2]
    if search_image.shape[0] < template_height or search_image.shape[1] < template_width:
        return None

    anchor_x = int(anchor.get("reference_point", {}).get("x", 0))
    anchor_y = int(anchor.get("reference_point", {}).get("y", 0))
    anchor_offset_x = anchor_x - tx0
    anchor_offset_y = anchor_y - ty0

    weights = 0.25 + (0.75 * (template_mask > 0).astype(np.float32))
    best_match: dict | None = None
    best_score = -1.0
    second_best_score = -1.0
    max_y = search_image.shape[0] - template_height
    max_x = search_image.shape[1] - template_width

    for offset_y in range(max_y + 1):
        for offset_x in range(max_x + 1):
            candidate_image = search_image[offset_y : offset_y + template_height, offset_x : offset_x + template_width]
            candidate_mask = search_mask[offset_y : offset_y + template_height, offset_x : offset_x + template_width]
            image_score = _weighted_correlation_score(template_image, candidate_image, weights, np)
            mask_score = _dice_overlap_score(template_mask, candidate_mask, np)
            score = (0.65 * image_score) + (0.35 * mask_score)

            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_match = {
                    "sample_point": (float(x0 + offset_x + anchor_offset_x), float(y0 + offset_y + anchor_offset_y)),
                    "confidence": float(score),
                    "method": "template_match",
                    "search_window": {
                        "x": int(x0),
                        "y": int(y0),
                        "width": int(x1 - x0),
                        "height": int(y1 - y0),
                    },
                    "template_bounds": {
                        "x": int(tx0),
                        "y": int(ty0),
                        "width": int(tx1 - tx0),
                        "height": int(ty1 - ty0),
                    },
                    "match_bounds": {
                        "x": int(x0 + offset_x),
                        "y": int(y0 + offset_y),
                        "width": int(template_width),
                        "height": int(template_height),
                    },
                    "image_score": float(image_score),
                    "mask_score": float(mask_score),
                }
            elif score > second_best_score:
                second_best_score = score

    if best_match is None:
        return None

    score_gap = max(0.0, best_score - max(0.0, second_best_score))
    uniqueness = max(0.0, min(1.0, score_gap * 4.0))
    best_match["confidence"] = float(max(0.0, min(1.0, (best_score * 0.8) + (uniqueness * 0.2))))
    best_match["score_gap"] = float(score_gap)
    return best_match


def _locate_sample_anchor_centroid(mask, anchor: dict, search_margin_px: int, np) -> dict:
    x0, y0, x1, y1 = _resolve_search_window(mask.shape[:2], anchor, search_margin_px)
    window = mask[y0:y1, x0:x1] > 0
    points = np.argwhere(window)
    if points.size == 0:
        return {
            "sample_point": None,
            "confidence": 0.0,
            "method": "centroid_fallback",
            "search_window": {
                "x": int(x0),
                "y": int(y0),
                "width": int(x1 - x0),
                "height": int(y1 - y0),
            },
        }

    centroid_y = float(points[:, 0].mean() + y0)
    centroid_x = float(points[:, 1].mean() + x0)
    window_area = max(1, (y1 - y0) * (x1 - x0))
    density = min(1.0, float(points.shape[0]) / float(window_area))
    span_x = float(points[:, 1].max() - points[:, 1].min() + 1)
    span_y = float(points[:, 0].max() - points[:, 0].min() + 1)
    compactness = 1.0 / (1.0 + ((span_x / max(1, x1 - x0)) + (span_y / max(1, y1 - y0))) / 2.0)
    confidence = max(0.05, min(0.75, 0.35 + (0.4 * density) + (0.25 * compactness)))
    return {
        "sample_point": (centroid_x, centroid_y),
        "confidence": float(confidence),
        "method": "centroid_fallback",
        "search_window": {
            "x": int(x0),
            "y": int(y0),
            "width": int(x1 - x0),
            "height": int(y1 - y0),
        },
    }


def _locate_sample_anchor(
    sample_mask,
    reference_mask,
    anchor: dict,
    search_margin_px: int,
    np,
    *,
    sample_registration_image=None,
    reference_registration_image=None,
) -> dict:
    template_match = _match_anchor_template(
        sample_mask,
        reference_mask,
        sample_registration_image,
        reference_registration_image,
        anchor,
        search_margin_px,
        np,
    )
    if template_match is not None and template_match.get("sample_point") is not None:
        return template_match
    return _locate_sample_anchor_centroid(sample_mask, anchor, search_margin_px, np)


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


def _apply_rigid_transform_to_image(image, angle_deg: float, shift_x: int, shift_y: int, cv2, np):
    if image is None:
        return None
    if not abs(angle_deg) > 1e-6 and not shift_x and not shift_y:
        return image
    if not hasattr(cv2, "getRotationMatrix2D") or not hasattr(cv2, "warpAffine"):
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    interpolation = getattr(cv2, "INTER_LINEAR", getattr(cv2, "INTER_NEAREST", 0))
    border_mode = getattr(cv2, "BORDER_CONSTANT", 0)
    return cv2.warpAffine(
        image,
        np.float32(matrix),
        (width, height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=0,
    )


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


def _compute_registration_confidence(
    reference_mask,
    sample_mask,
    reference_registration_image,
    aligned_registration_image,
    np,
) -> float:
    mask_confidence = _compute_mask_overlap_confidence(reference_mask, sample_mask, np)
    if reference_registration_image is None or aligned_registration_image is None:
        return float(mask_confidence)

    weights = np.ones_like(reference_registration_image, dtype=np.float32)
    image_confidence = _weighted_correlation_score(
        reference_registration_image,
        aligned_registration_image,
        weights,
        np,
    )
    return float((0.55 * image_confidence) + (0.45 * mask_confidence))


def _refine_registration_with_images(
    sample_mask,
    reference_mask,
    coarse_angle_deg: float,
    coarse_shift_x: int,
    coarse_shift_y: int,
    alignment_cfg: dict,
    registration_cfg: dict,
    cv2,
    np,
    *,
    sample_registration_image=None,
    reference_registration_image=None,
) -> tuple[object, float, int, int, float, float | None]:
    coarse_aligned_mask = _apply_rigid_transform_to_mask(
        sample_mask,
        coarse_angle_deg,
        coarse_shift_x,
        coarse_shift_y,
        cv2,
        np,
    )
    coarse_aligned_registration_image = _apply_rigid_transform_to_image(
        sample_registration_image,
        coarse_angle_deg,
        coarse_shift_x,
        coarse_shift_y,
        cv2,
        np,
    )
    best_angle_deg = float(coarse_angle_deg)
    best_shift_x = int(coarse_shift_x)
    best_shift_y = int(coarse_shift_y)
    best_aligned_mask = coarse_aligned_mask
    best_confidence = _compute_registration_confidence(
        reference_mask,
        coarse_aligned_mask,
        reference_registration_image,
        coarse_aligned_registration_image,
        np,
    )
    best_residual = _compute_mask_centroid_residual(reference_mask, coarse_aligned_mask, cv2, np)

    for candidate_angle_deg, candidate_shift_x, candidate_shift_y in _build_rigid_refinement_candidates(
        coarse_angle_deg,
        coarse_shift_x,
        coarse_shift_y,
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
        candidate_registration_image = _apply_rigid_transform_to_image(
            sample_registration_image,
            candidate_angle_deg,
            candidate_shift_x,
            candidate_shift_y,
            cv2,
            np,
        )
        candidate_confidence = _compute_registration_confidence(
            reference_mask,
            candidate_mask,
            reference_registration_image,
            candidate_registration_image,
            np,
        )
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

    return best_aligned_mask, best_angle_deg, best_shift_x, best_shift_y, float(best_confidence), best_residual


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


def _register_with_anchor_translation(
    sample_mask,
    reference_mask,
    alignment_cfg: dict,
    registration_cfg: dict,
    cv2,
    np,
    *,
    sample_registration_image=None,
    reference_registration_image=None,
) -> RegistrationResult:
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
    anchor_match = _locate_sample_anchor(
        sample_mask,
        reference_mask,
        anchor,
        registration_cfg["search_margin_px"],
        np,
        sample_registration_image=sample_registration_image,
        reference_registration_image=reference_registration_image,
    )
    sample_anchor = anchor_match.get("sample_point")
    confidence = float(anchor_match.get("confidence", 0.0))
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
            observed_anchors=[
                {
                    "anchor_id": anchor["anchor_id"],
                    "reference_point": dict(anchor["reference_point"]),
                    "sample_point": None,
                    "confidence": float(anchor_match.get("confidence", 0.0)),
                    "localization_method": anchor_match.get("method"),
                    "search_window": anchor_match.get("search_window"),
                }
            ],
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
                "confidence": float(confidence),
                "localization_method": anchor_match.get("method"),
                "search_window": anchor_match.get("search_window"),
                "template_bounds": anchor_match.get("template_bounds"),
                "match_bounds": anchor_match.get("match_bounds"),
                "image_score": anchor_match.get("image_score"),
                "mask_score": anchor_match.get("mask_score"),
                "score_gap": anchor_match.get("score_gap"),
            }
        ],
    ))


def _register_with_anchor_pair(
    sample_mask,
    reference_mask,
    alignment_cfg: dict,
    registration_cfg: dict,
    cv2,
    np,
    *,
    sample_registration_image=None,
    reference_registration_image=None,
) -> RegistrationResult:
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
    primary_match = _locate_sample_anchor(
        sample_mask,
        reference_mask,
        primary_anchor,
        registration_cfg["search_margin_px"],
        np,
        sample_registration_image=sample_registration_image,
        reference_registration_image=reference_registration_image,
    )
    secondary_match = _locate_sample_anchor(
        sample_mask,
        reference_mask,
        secondary_anchor,
        registration_cfg["search_margin_px"],
        np,
        sample_registration_image=sample_registration_image,
        reference_registration_image=reference_registration_image,
    )
    primary_sample = primary_match.get("sample_point")
    secondary_sample = secondary_match.get("sample_point")
    primary_confidence = float(primary_match.get("confidence", 0.0))
    secondary_confidence = float(secondary_match.get("confidence", 0.0))
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
                    "confidence": float(primary_confidence),
                    "localization_method": primary_match.get("method"),
                    "search_window": primary_match.get("search_window"),
                },
                {
                    "anchor_id": secondary_anchor["anchor_id"],
                    "reference_point": dict(secondary_anchor["reference_point"]),
                    "sample_point": None if secondary_sample is None else {"x": float(secondary_sample[0]), "y": float(secondary_sample[1])},
                    "confidence": float(secondary_confidence),
                    "localization_method": secondary_match.get("method"),
                    "search_window": secondary_match.get("search_window"),
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
                "confidence": float(primary_confidence),
                "localization_method": primary_match.get("method"),
                "search_window": primary_match.get("search_window"),
                "template_bounds": primary_match.get("template_bounds"),
                "match_bounds": primary_match.get("match_bounds"),
                "image_score": primary_match.get("image_score"),
                "mask_score": primary_match.get("mask_score"),
                "score_gap": primary_match.get("score_gap"),
            },
            {
                "anchor_id": secondary_anchor["anchor_id"],
                "reference_point": dict(secondary_anchor["reference_point"]),
                "sample_point": {"x": float(secondary_sample[0]), "y": float(secondary_sample[1])},
                "transformed_point": {"x": float(transformed_secondary[0]), "y": float(transformed_secondary[1])},
                "confidence": float(secondary_confidence),
                "localization_method": secondary_match.get("method"),
                "search_window": secondary_match.get("search_window"),
                "template_bounds": secondary_match.get("template_bounds"),
                "match_bounds": secondary_match.get("match_bounds"),
                "image_score": secondary_match.get("image_score"),
                "mask_score": secondary_match.get("mask_score"),
                "score_gap": secondary_match.get("score_gap"),
            },
        ],
    ))


def register_sample_mask(
    sample_mask,
    reference_mask,
    alignment_cfg: dict,
    cv2,
    np,
    align_sample_mask_fn,
    *,
    sample_registration_image=None,
    reference_registration_image=None,
) -> RegistrationResult:
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
        return _register_with_anchor_translation(
            sample_mask,
            reference_mask,
            alignment_cfg,
            registration_cfg,
            cv2,
            np,
            sample_registration_image=sample_registration_image,
            reference_registration_image=reference_registration_image,
        )
    elif runtime_mode == "anchor_pair":
        return _register_with_anchor_pair(
            sample_mask,
            reference_mask,
            alignment_cfg,
            registration_cfg,
            cv2,
            np,
            sample_registration_image=sample_registration_image,
            reference_registration_image=reference_registration_image,
        )
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
        if sample_registration_image is not None and reference_registration_image is not None:
            aligned_mask, angle_deg, shift_x, shift_y, confidence, residual = _refine_registration_with_images(
                sample_mask,
                reference_mask,
                float(angle_deg),
                int(shift_x),
                int(shift_y),
                alignment_cfg,
                registration_cfg,
                cv2,
                np,
                sample_registration_image=sample_registration_image,
                reference_registration_image=reference_registration_image,
            )
            quality = {
                "confidence": float(confidence),
                "mean_residual_px": None if residual is None else float(residual),
            }
        else:
            quality = {
                "confidence": _compute_mask_overlap_confidence(reference_mask, aligned_mask, np),
                "mean_residual_px": _compute_mask_centroid_residual(reference_mask, aligned_mask, cv2, np),
            }
        if requested_strategy != runtime_mode:
            fallback_reason = (
                f"Requested registration strategy '{requested_strategy}' is staged but runtime is using '{runtime_mode}'."
            )

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
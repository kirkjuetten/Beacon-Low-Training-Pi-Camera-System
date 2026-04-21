from __future__ import annotations

from itertools import combinations
import math


DEFAULT_MOLDED_PART_FEATURE_FAMILIES = ("isolated_centroid", "paired_centroid")


def _connected_components_numpy(binary_mask, np_module) -> list[dict]:
    height, width = binary_mask.shape[:2]
    visited = np_module.zeros_like(binary_mask, dtype=bool)
    components: list[dict] = []

    for start_y in range(height):
        for start_x in range(width):
            if not bool(binary_mask[start_y, start_x]) or bool(visited[start_y, start_x]):
                continue

            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        if delta_y == 0 and delta_x == 0:
                            continue
                        next_y = y + delta_y
                        next_x = x + delta_x
                        if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                            continue
                        if not bool(binary_mask[next_y, next_x]) or bool(visited[next_y, next_x]):
                            continue
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))

            ys = [pixel[0] for pixel in pixels]
            xs = [pixel[1] for pixel in pixels]
            x0 = min(xs)
            x1 = max(xs) + 1
            y0 = min(ys)
            y1 = max(ys) + 1
            components.append(
                {
                    "area": int(len(pixels)),
                    "bbox": {"x": int(x0), "y": int(y0), "width": int(x1 - x0), "height": int(y1 - y0)},
                    "center": {"x": float(sum(xs) / len(xs)), "y": float(sum(ys) / len(ys))},
                }
            )

    return components


def _connected_components(mask, cv2, np_module) -> list[dict]:
    binary_mask = (mask > 0).astype(np_module.uint8)
    if int(binary_mask.sum()) == 0:
        return []

    if not hasattr(cv2, "connectedComponentsWithStats"):
        return _connected_components_numpy(binary_mask, np_module)

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
    components: list[dict] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        center_x, center_y = centroids[index]
        components.append(
            {
                "area": area,
                "bbox": {"x": x, "y": y, "width": width, "height": height},
                "center": {"x": float(center_x), "y": float(center_y)},
            }
        )
    return components


def _reference_position_feature_components(reference_required, cv2, np_module) -> list[dict]:
    components = _connected_components(reference_required, cv2, np_module)
    if len(components) < 2:
        return []

    largest_area = max(component["area"] for component in components)
    min_area = max(20, int(round(largest_area * 0.0002)))
    max_area = max(min_area, min(5000, int(round(largest_area * 0.05))))

    candidates = [
        component
        for component in components
        if min_area <= int(component["area"]) <= max_area
    ]
    candidates.sort(key=lambda component: (component["center"]["x"], component["center"]["y"]))
    return candidates[:4]


def _sort_components(components: list[dict]) -> list[dict]:
    return sorted(components, key=lambda component: (float(component["center"]["x"]), float(component["center"]["y"])))


def _union_bbox(components: list[dict]) -> dict:
    x0 = min(int(component["bbox"]["x"]) for component in components)
    y0 = min(int(component["bbox"]["y"]) for component in components)
    x1 = max(int(component["bbox"]["x"]) + int(component["bbox"]["width"]) for component in components)
    y1 = max(int(component["bbox"]["y"]) + int(component["bbox"]["height"]) for component in components)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _window_details(mask_shape: tuple[int, int], bbox: dict) -> tuple[tuple[int, int, int, int], dict]:
    x0, y0, x1, y1 = _window_bounds(mask_shape, bbox)
    return (x0, y0, x1, y1), {
        "x": int(x0),
        "y": int(y0),
        "width": int(x1 - x0),
        "height": int(y1 - y0),
    }


def _center_from_components(components: list[dict]) -> dict:
    return {
        "x": float(sum(float(component["center"]["x"]) for component in components) / len(components)),
        "y": float(sum(float(component["center"]["y"]) for component in components) / len(components)),
    }


def _component_spacing_px(first: dict, second: dict) -> float:
    return float(
        math.hypot(
            float(second["center"]["x"]) - float(first["center"]["x"]),
            float(second["center"]["y"]) - float(first["center"]["y"]),
        )
    )


def _component_angle_deg(first: dict, second: dict) -> float:
    return float(
        math.degrees(
            math.atan2(
                float(second["center"]["y"]) - float(first["center"]["y"]),
                float(second["center"]["x"]) - float(first["center"]["x"]),
            )
        )
    )


def _build_feature_measurement(
    *,
    feature_key: str,
    feature_label: str,
    feature_family: str,
    feature_type: str,
    reference_center: dict,
    expected_sample_window: dict,
    reference_area_px: int,
    sample_detected: bool,
    failure_cause: str,
    observed_center_reference: dict | None = None,
    observed_area_px: int | None = None,
    dx_px: float | None = None,
    dy_px: float | None = None,
    radial_offset_px: float | None = None,
    center_offset_px: float | None = None,
    extra_fields: dict | None = None,
) -> dict:
    measurement = {
        "feature_key": feature_key,
        "feature_label": feature_label,
        "feature_family": feature_family,
        "feature_type": feature_type,
        "measurement_frame": "datum",
        "sample_detected": bool(sample_detected),
        "failure_cause": failure_cause,
        "reference_center": reference_center,
        "expected_sample_window": expected_sample_window,
        "reference_area_px": int(reference_area_px),
        "observed_area_px": observed_area_px,
        "observed_center_reference": observed_center_reference,
        "dx_px": dx_px,
        "dy_px": dy_px,
        "radial_offset_px": radial_offset_px,
        "center_offset_px": center_offset_px,
    }
    if extra_fields:
        measurement.update(extra_fields)
    return measurement


def _window_bounds(mask_shape: tuple[int, int], bbox: dict) -> tuple[int, int, int, int]:
    height, width = mask_shape[:2]
    pad_x = max(4, int(bbox.get("width", 0)))
    pad_y = max(4, int(bbox.get("height", 0)))
    x0 = max(0, int(bbox.get("x", 0)) - pad_x)
    y0 = max(0, int(bbox.get("y", 0)) - pad_y)
    x1 = min(width, int(bbox.get("x", 0)) + int(bbox.get("width", 0)) + pad_x)
    y1 = min(height, int(bbox.get("y", 0)) + int(bbox.get("height", 0)) + pad_y)
    return x0, y0, x1, y1


def _closest_observed_component(window_mask, reference_center: dict, window_origin: tuple[int, int], cv2, np_module) -> dict | None:
    components = _connected_components(window_mask, cv2, np_module)
    if not components:
        return None

    origin_x, origin_y = window_origin
    ref_x = float(reference_center.get("x", 0.0)) - float(origin_x)
    ref_y = float(reference_center.get("y", 0.0)) - float(origin_y)

    def _rank(component: dict) -> tuple[float, float]:
        center = component["center"]
        distance = math.hypot(float(center["x"]) - ref_x, float(center["y"]) - ref_y)
        return (distance, -float(component["area"]))

    best = min(components, key=_rank)
    return {
        "area": int(best["area"]),
        "bbox": {
            "x": int(best["bbox"]["x"]) + origin_x,
            "y": int(best["bbox"]["y"]) + origin_y,
            "width": int(best["bbox"]["width"]),
            "height": int(best["bbox"]["height"]),
        },
        "center": {
            "x": float(best["center"]["x"]) + float(origin_x),
            "y": float(best["center"]["y"]) + float(origin_y),
        },
    }


def _reference_paired_centroid_features(reference_features: list[dict]) -> list[dict]:
    ordered_features = _sort_components(reference_features)
    paired_features: list[dict] = []
    for index in range(len(ordered_features) - 1):
        members = [ordered_features[index], ordered_features[index + 1]]
        paired_features.append(
            {
                "pair_index": index,
                "members": members,
                "bbox": _union_bbox(members),
                "center": _center_from_components(members),
                "reference_area_px": int(sum(int(member["area"]) for member in members)),
                "reference_spacing_px": _component_spacing_px(members[0], members[1]),
                "reference_angle_deg": _component_angle_deg(members[0], members[1]),
            }
        )
    return paired_features[:3]


def _closest_observed_pair(window_mask, reference_members: list[dict], window_origin: tuple[int, int], cv2, np_module) -> dict | None:
    observed_components = _sort_components(_connected_components(window_mask, cv2, np_module))
    if len(observed_components) < 2:
        return None

    origin_x, origin_y = window_origin
    local_reference_members = []
    for member in _sort_components(reference_members):
        local_reference_members.append(
            {
                "x": float(member["center"]["x"]) - float(origin_x),
                "y": float(member["center"]["y"]) - float(origin_y),
            }
        )

    reference_pair_center = {
        "x": float(sum(member["x"] for member in local_reference_members) / len(local_reference_members)),
        "y": float(sum(member["y"] for member in local_reference_members) / len(local_reference_members)),
    }
    reference_spacing_px = float(
        math.hypot(
            local_reference_members[1]["x"] - local_reference_members[0]["x"],
            local_reference_members[1]["y"] - local_reference_members[0]["y"],
        )
    )

    best_pair = None
    best_rank = None
    for first_component, second_component in combinations(observed_components, 2):
        candidate_members = _sort_components([first_component, second_component])
        member_distance = 0.0
        for reference_member, observed_member in zip(local_reference_members, candidate_members):
            member_distance += math.hypot(
                float(observed_member["center"]["x"]) - reference_member["x"],
                float(observed_member["center"]["y"]) - reference_member["y"],
            )

        candidate_pair_center = _center_from_components(candidate_members)
        pair_center_distance = math.hypot(
            float(candidate_pair_center["x"]) - reference_pair_center["x"],
            float(candidate_pair_center["y"]) - reference_pair_center["y"],
        )
        candidate_spacing_px = _component_spacing_px(candidate_members[0], candidate_members[1])
        spacing_delta_px = abs(candidate_spacing_px - reference_spacing_px)
        rank = (pair_center_distance, member_distance, spacing_delta_px)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_pair = candidate_members

    if best_pair is None:
        return None

    observed_pair_center = _center_from_components(best_pair)
    return {
        "members": [
            {
                "area": int(member["area"]),
                "bbox": {
                    "x": int(member["bbox"]["x"]) + origin_x,
                    "y": int(member["bbox"]["y"]) + origin_y,
                    "width": int(member["bbox"]["width"]),
                    "height": int(member["bbox"]["height"]),
                },
                "center": {
                    "x": float(member["center"]["x"]) + float(origin_x),
                    "y": float(member["center"]["y"]) + float(origin_y),
                },
            }
            for member in best_pair
        ],
        "center": {
            "x": float(observed_pair_center["x"]) + float(origin_x),
            "y": float(observed_pair_center["y"]) + float(origin_y),
        },
        "observed_area_px": int(sum(int(member["area"]) for member in best_pair)),
        "observed_spacing_px": _component_spacing_px(best_pair[0], best_pair[1]),
        "observed_angle_deg": _component_angle_deg(best_pair[0], best_pair[1]),
    }


def _extract_isolated_centroid_measurements(reference_features: list[dict], sample_datum_mask, cv2, np_module) -> list[dict]:
    feature_measurements: list[dict] = []
    sample_white = sample_datum_mask > 0
    for index, reference_feature in enumerate(_sort_components(reference_features)):
        reference_center = dict(reference_feature["center"])
        (_x0, _y0, _x1, _y1), expected_sample_window = _window_details(sample_datum_mask.shape[:2], dict(reference_feature["bbox"]))
        x0 = expected_sample_window["x"]
        y0 = expected_sample_window["y"]
        x1 = x0 + expected_sample_window["width"]
        y1 = y0 + expected_sample_window["height"]
        window_mask = sample_white[y0:y1, x0:x1]
        observed_component = _closest_observed_component(window_mask, reference_center, (x0, y0), cv2, np_module)

        feature_key = f"isolated_feature_{index + 1}"
        feature_label = f"Isolated Feature {index + 1}"
        if observed_component is None:
            feature_measurements.append(
                _build_feature_measurement(
                    feature_key=feature_key,
                    feature_label=feature_label,
                    feature_family="isolated_centroid",
                    feature_type="isolated_centroid_position",
                    reference_center=reference_center,
                    expected_sample_window=expected_sample_window,
                    reference_area_px=int(reference_feature["area"]),
                    sample_detected=False,
                    failure_cause="feature_not_found",
                )
            )
            continue

        observed_center = dict(observed_component["center"])
        dx_px = float(observed_center["x"] - reference_center["x"])
        dy_px = float(observed_center["y"] - reference_center["y"])
        radial_offset_px = float(math.hypot(dx_px, dy_px))
        feature_measurements.append(
            _build_feature_measurement(
                feature_key=feature_key,
                feature_label=feature_label,
                feature_family="isolated_centroid",
                feature_type="isolated_centroid_position",
                reference_center=reference_center,
                expected_sample_window=expected_sample_window,
                reference_area_px=int(reference_feature["area"]),
                sample_detected=True,
                failure_cause="feature_position",
                observed_center_reference=observed_center,
                observed_area_px=int(observed_component["area"]),
                dx_px=dx_px,
                dy_px=dy_px,
                radial_offset_px=radial_offset_px,
                center_offset_px=radial_offset_px,
            )
        )
    return feature_measurements


def _extract_paired_centroid_measurements(reference_features: list[dict], sample_datum_mask, cv2, np_module) -> list[dict]:
    paired_features = _reference_paired_centroid_features(reference_features)
    if not paired_features:
        return []

    feature_measurements: list[dict] = []
    sample_white = sample_datum_mask > 0
    for paired_feature in paired_features:
        pair_index = int(paired_feature["pair_index"])
        reference_center = dict(paired_feature["center"])
        members = list(paired_feature["members"])
        (_x0, _y0, _x1, _y1), expected_sample_window = _window_details(sample_datum_mask.shape[:2], dict(paired_feature["bbox"]))
        x0 = expected_sample_window["x"]
        y0 = expected_sample_window["y"]
        x1 = x0 + expected_sample_window["width"]
        y1 = y0 + expected_sample_window["height"]
        window_mask = sample_white[y0:y1, x0:x1]
        observed_pair = _closest_observed_pair(window_mask, members, (x0, y0), cv2, np_module)

        feature_key = f"paired_feature_{pair_index + 1}"
        feature_label = f"Paired Feature {pair_index + 1}"
        extra_fields = {
            "reference_centers": [dict(member["center"]) for member in _sort_components(members)],
            "feature_members": [f"isolated_feature_{pair_index + 1}", f"isolated_feature_{pair_index + 2}"],
            "pair_spacing_reference_px": float(paired_feature["reference_spacing_px"]),
            "pair_spacing_observed_px": None,
            "pair_spacing_delta_px": None,
            "pair_angle_reference_deg": float(paired_feature["reference_angle_deg"]),
            "pair_angle_observed_deg": None,
            "pair_angle_delta_deg": None,
        }
        if observed_pair is None:
            feature_measurements.append(
                _build_feature_measurement(
                    feature_key=feature_key,
                    feature_label=feature_label,
                    feature_family="paired_centroid",
                    feature_type="paired_centroid_position",
                    reference_center=reference_center,
                    expected_sample_window=expected_sample_window,
                    reference_area_px=int(paired_feature["reference_area_px"]),
                    sample_detected=False,
                    failure_cause="feature_not_found",
                    extra_fields=extra_fields,
                )
            )
            continue

        observed_center = dict(observed_pair["center"])
        dx_px = float(observed_center["x"] - reference_center["x"])
        dy_px = float(observed_center["y"] - reference_center["y"])
        radial_offset_px = float(math.hypot(dx_px, dy_px))
        pair_spacing_observed_px = float(observed_pair["observed_spacing_px"])
        pair_spacing_delta_px = float(abs(pair_spacing_observed_px - float(paired_feature["reference_spacing_px"])))
        pair_angle_observed_deg = float(observed_pair["observed_angle_deg"])
        pair_angle_delta_deg = float(abs(pair_angle_observed_deg - float(paired_feature["reference_angle_deg"])))
        extra_fields.update(
            {
                "observed_centers_reference": [dict(member["center"]) for member in observed_pair["members"]],
                "pair_spacing_observed_px": pair_spacing_observed_px,
                "pair_spacing_delta_px": pair_spacing_delta_px,
                "pair_angle_observed_deg": pair_angle_observed_deg,
                "pair_angle_delta_deg": pair_angle_delta_deg,
            }
        )
        feature_measurements.append(
            _build_feature_measurement(
                feature_key=feature_key,
                feature_label=feature_label,
                feature_family="paired_centroid",
                feature_type="paired_centroid_position",
                reference_center=reference_center,
                expected_sample_window=expected_sample_window,
                reference_area_px=int(paired_feature["reference_area_px"]),
                sample_detected=True,
                failure_cause="feature_position",
                observed_center_reference=observed_center,
                observed_area_px=int(observed_pair["observed_area_px"]),
                dx_px=dx_px,
                dy_px=dy_px,
                radial_offset_px=radial_offset_px,
                center_offset_px=radial_offset_px,
                extra_fields=extra_fields,
            )
        )
    return feature_measurements


def summarize_feature_measurements(feature_measurements: list[dict]) -> dict | None:
    if not feature_measurements:
        return None

    def _rank(entry: dict) -> tuple[int, float, float]:
        if not entry.get("sample_detected", False):
            return (1, float("inf"), float("inf"))
        return (
            0,
            float(entry.get("radial_offset_px") or entry.get("center_offset_px") or 0.0),
            float(entry.get("pair_spacing_delta_px") or 0.0),
        )

    worst_feature = max(feature_measurements, key=_rank)
    return {
        "feature_key": worst_feature.get("feature_key"),
        "feature_label": worst_feature.get("feature_label"),
        "feature_family": worst_feature.get("feature_family"),
        "feature_type": worst_feature.get("feature_type"),
        "measurement_frame": worst_feature.get("measurement_frame"),
        "feature_count": len(feature_measurements),
        "sample_detected": bool(worst_feature.get("sample_detected", False)),
        "failure_cause": worst_feature.get("failure_cause"),
        "reference_center": worst_feature.get("reference_center"),
        "reference_centers": worst_feature.get("reference_centers"),
        "expected_sample_window": worst_feature.get("expected_sample_window"),
        "observed_center_reference": worst_feature.get("observed_center_reference"),
        "observed_centers_reference": worst_feature.get("observed_centers_reference"),
        "reference_area_px": worst_feature.get("reference_area_px"),
        "observed_area_px": worst_feature.get("observed_area_px"),
        "dx_px": worst_feature.get("dx_px"),
        "dy_px": worst_feature.get("dy_px"),
        "radial_offset_px": worst_feature.get("radial_offset_px"),
        "center_offset_px": worst_feature.get("center_offset_px"),
        "feature_members": worst_feature.get("feature_members"),
        "pair_spacing_reference_px": worst_feature.get("pair_spacing_reference_px"),
        "pair_spacing_observed_px": worst_feature.get("pair_spacing_observed_px"),
        "pair_spacing_delta_px": worst_feature.get("pair_spacing_delta_px"),
        "pair_angle_reference_deg": worst_feature.get("pair_angle_reference_deg"),
        "pair_angle_observed_deg": worst_feature.get("pair_angle_observed_deg"),
        "pair_angle_delta_deg": worst_feature.get("pair_angle_delta_deg"),
    }


def extract_molded_part_feature_measurements(
    reference_required,
    sample_datum_mask,
    feature_families,
    cv2,
    np_module,
) -> tuple[list[dict], dict | None]:
    reference_features = _reference_position_feature_components(reference_required, cv2, np_module)
    if not reference_features:
        return [], None

    requested_families: list[str] = []
    for family_name in feature_families or DEFAULT_MOLDED_PART_FEATURE_FAMILIES:
        normalized_name = str(family_name or "").strip().lower()
        if normalized_name in DEFAULT_MOLDED_PART_FEATURE_FAMILIES and normalized_name not in requested_families:
            requested_families.append(normalized_name)
    if not requested_families:
        requested_families = list(DEFAULT_MOLDED_PART_FEATURE_FAMILIES)

    feature_measurements: list[dict] = []
    for family_name in requested_families:
        if family_name == "isolated_centroid":
            feature_measurements.extend(
                _extract_isolated_centroid_measurements(reference_features, sample_datum_mask, cv2, np_module)
            )
        elif family_name == "paired_centroid":
            feature_measurements.extend(
                _extract_paired_centroid_measurements(reference_features, sample_datum_mask, cv2, np_module)
            )

    return feature_measurements, summarize_feature_measurements(feature_measurements)


def extract_localized_feature_position_features(reference_required, sample_datum_mask, cv2, np_module) -> tuple[list[dict], dict | None]:
    """Compatibility wrapper for older generic callers; returns the isolated-centroid family."""
    return extract_molded_part_feature_measurements(
        reference_required,
        sample_datum_mask,
        ["isolated_centroid"],
        cv2,
        np_module,
    )


def extract_light_pipe_position_features(reference_required, sample_datum_mask, cv2, np_module) -> tuple[list[dict], dict | None]:
    """Compatibility wrapper for older callers; returns the isolated-centroid family contract."""
    return extract_localized_feature_position_features(reference_required, sample_datum_mask, cv2, np_module)
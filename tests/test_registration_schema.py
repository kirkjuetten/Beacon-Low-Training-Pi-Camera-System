from inspection_system.app.registration_schema import (
    build_alignment_metadata,
    get_registration_commissioning_config,
    get_registration_config,
    normalize_registration_anchors,
)


def test_get_registration_config_uses_legacy_alignment_mode_as_default_strategy() -> None:
    config = {
        "alignment": {
            "mode": "moments",
        }
    }

    registration = get_registration_config(config)

    assert registration["strategy"] == "moments"
    assert registration["transform_model"] == "rigid"
    assert registration["anchor_mode"] == "none"
    assert registration["datum_frame"]["origin"] == "roi_top_left"


def test_get_registration_config_normalizes_anchor_definitions() -> None:
    config = {
        "alignment": {
            "mode": "moments",
            "registration": {
                "strategy": "anchor_pair",
                "transform_model": "similarity",
                "anchor_mode": "pair",
                "subpixel_refinement": "phase_correlation",
                "search_margin_px": "28",
                "quality_gates": {
                    "min_confidence": "0.85",
                    "max_mean_residual_px": "1.4",
                },
                "datum_frame": {
                    "origin": "anchor_primary",
                    "orientation": "anchor_pair",
                },
                "anchors": [
                    {
                        "id": "left_pad",
                        "kind": "corner",
                        "x": 12,
                        "y": 18,
                        "search_window": {"x": 5, "y": 6, "width": 30, "height": 40},
                    }
                ],
            },
        }
    }

    registration = get_registration_config(config)

    assert registration["strategy"] == "anchor_pair"
    assert registration["transform_model"] == "similarity"
    assert registration["anchor_mode"] == "pair"
    assert registration["subpixel_refinement"] == "phase_correlation"
    assert registration["search_margin_px"] == 28
    assert registration["quality_gates"]["min_confidence"] == 0.85
    assert registration["quality_gates"]["max_mean_residual_px"] == 1.4
    assert registration["datum_frame"]["origin"] == "anchor_primary"
    assert registration["anchors"] == [
        {
            "anchor_id": "left_pad",
            "label": "Left Pad",
            "kind": "corner",
            "enabled": True,
            "reference_point": {"x": 12, "y": 18},
            "search_window": {"x": 5, "y": 6, "width": 30, "height": 40},
        }
    ]


def test_build_alignment_metadata_embeds_registration_contract() -> None:
    config = {
        "alignment": {
            "enabled": True,
            "mode": "moments",
            "tolerance_profile": "strict",
            "max_angle_deg": 2.0,
            "max_shift_x": 7,
            "max_shift_y": 8,
            "registration": {
                "strategy": "anchor_pair",
                "transform_model": "similarity",
            },
        }
    }

    metadata = build_alignment_metadata(config)

    assert metadata["enabled"] is True
    assert metadata["mode"] == "moments"
    assert metadata["tolerance_profile"] == "strict"
    assert metadata["max_angle_deg"] == 2.0
    assert metadata["max_shift_x"] == 7
    assert metadata["max_shift_y"] == 8
    assert metadata["registration"]["strategy"] == "anchor_pair"
    assert metadata["registration"]["transform_model"] == "similarity"


def test_get_registration_commissioning_config_defaults_flags_off() -> None:
    commissioning = get_registration_commissioning_config({"alignment": {"registration": {}}})

    assert commissioning == {
        "datum_confirmed": False,
        "expected_transform_confirmed": False,
    }


def test_normalize_registration_anchors_coerces_sparse_entries() -> None:
    anchors = normalize_registration_anchors(
        [
            {
                "id": "left_pad",
                "x": 12,
                "y": 18,
                "search_window": {"x": 5, "y": 6, "width": 30, "height": 40},
            }
        ]
    )

    assert anchors == [
        {
            "anchor_id": "left_pad",
            "label": "Left Pad",
            "kind": "feature",
            "enabled": True,
            "reference_point": {"x": 12, "y": 18},
            "search_window": {"x": 5, "y": 6, "width": 30, "height": 40},
        }
    ]
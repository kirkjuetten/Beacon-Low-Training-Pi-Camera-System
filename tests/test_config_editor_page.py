from inspection_system.app.config_editor_page import apply_registration_setup, build_registration_setup_values


def test_build_registration_setup_values_reads_commissioning_and_transform_fields() -> None:
    values = build_registration_setup_values(
        {
            "alignment": {
                "mode": "anchor_pair",
                "max_angle_deg": 1.4,
                "max_shift_x": 6,
                "max_shift_y": 5,
                "registration": {
                    "strategy": "anchor_pair",
                    "transform_model": "similarity",
                    "anchor_mode": "pair",
                    "subpixel_refinement": "template",
                    "search_margin_px": 36,
                    "datum_frame": {
                        "origin": "anchor_primary",
                        "orientation": "anchor_pair",
                    },
                    "commissioning": {
                        "datum_confirmed": True,
                        "expected_transform_confirmed": False,
                    },
                },
            }
        }
    )

    assert values["alignment.mode"] == "anchor_pair"
    assert values["alignment.registration.transform_model"] == "similarity"
    assert values["alignment.registration.search_margin_px"] == 36
    assert values["alignment.max_angle_deg"] == 1.4
    assert values["alignment.registration.commissioning.datum_confirmed"] is True
    assert values["alignment.registration.commissioning.expected_transform_confirmed"] is False


def test_apply_registration_setup_updates_anchor_and_commissioning_config() -> None:
    updated = apply_registration_setup(
        {"alignment": {"registration": {}}},
        {
            "alignment.mode": "anchor_pair",
            "alignment.registration.strategy": "anchor_pair",
            "alignment.registration.transform_model": "similarity",
            "alignment.registration.anchor_mode": "pair",
            "alignment.registration.subpixel_refinement": "template",
            "alignment.registration.search_margin_px": "32",
            "alignment.registration.datum_frame.origin": "anchor_primary",
            "alignment.registration.datum_frame.orientation": "anchor_pair",
            "alignment.max_angle_deg": "1.3",
            "alignment.max_shift_x": "6",
            "alignment.max_shift_y": "4",
            "alignment.registration.commissioning.datum_confirmed": True,
            "alignment.registration.commissioning.expected_transform_confirmed": True,
        },
        [
            {
                "anchor_id": "left_pad",
                "label": "Left Pad",
                "enabled": True,
                "reference_point": {"x": 12, "y": 18},
                "search_window": {"x": 5, "y": 6, "width": 30, "height": 40},
            },
            {
                "anchor_id": "right_pad",
                "label": "Right Pad",
                "enabled": True,
                "reference_point": {"x": 44, "y": 18},
                "search_window": {"x": 36, "y": 6, "width": 30, "height": 40},
            },
        ],
    )

    assert updated["alignment"]["mode"] == "anchor_pair"
    assert updated["alignment"]["max_angle_deg"] == 1.3
    assert updated["alignment"]["max_shift_x"] == 6
    assert updated["alignment"]["registration"]["strategy"] == "anchor_pair"
    assert updated["alignment"]["registration"]["transform_model"] == "similarity"
    assert updated["alignment"]["registration"]["datum_frame"]["origin"] == "anchor_primary"
    assert updated["alignment"]["registration"]["anchors"][0]["anchor_id"] == "left_pad"
    assert updated["alignment"]["registration"]["commissioning"]["datum_confirmed"] is True
    assert updated["alignment"]["registration"]["commissioning"]["expected_transform_confirmed"] is True
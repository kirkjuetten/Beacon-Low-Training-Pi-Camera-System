from inspection_system.app.config_editor_page import (
    apply_io_setup,
    apply_registration_setup,
    build_io_setup_values,
    build_registration_setup_values,
    format_io_setup_summary,
)


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


def test_build_io_setup_values_reads_runtime_and_hook_defaults() -> None:
    values = build_io_setup_values(
        {
            "io": {
                "mode": "modbus",
                "indicator_target": "relay",
                "modbus": {
                    "enabled": True,
                    "port": "/dev/ttyUSB0",
                    "baud": 9600,
                    "parity": "N",
                    "stopbits": 1,
                    "bytesize": 8,
                    "timeout_s": 1.0,
                    "slave_id": 1,
                },
                "relay": {"slave_id": 2, "pass_channel": 0, "fail_channel": 1},
                "trigger": {"enabled": True, "channel": 0, "debounce_ms": 30},
                "inputs": {
                    "part_in_nest": {"enabled": True, "channel": 1, "active_high": True},
                },
                "outputs": {
                    "review_needed": {"enabled": True, "channel": 2, "pulse_ms": 1200},
                },
                "pass_pulse_ms": 3000,
                "fail_pulse_ms": 2500,
            }
        }
    )

    assert values["io.mode"] == "modbus"
    assert values["io.modbus.enabled"] is True
    assert values["io.inputs.trigger_capture.enabled"] is True
    assert values["io.inputs.trigger_capture.channel"] == 0
    assert values["io.inputs.part_in_nest.enabled"] is True
    assert values["io.inputs.part_in_nest.channel"] == 1
    assert values["io.outputs.inspection_pass.channel"] == "0"
    assert values["io.outputs.inspection_fail.channel"] == "1"
    assert values["io.outputs.review_needed.channel"] == "2"
    assert values["io.outputs.inspection_fail.pulse_ms"] == 2500


def test_apply_io_setup_updates_runtime_compatible_and_hook_fields() -> None:
    updated = apply_io_setup(
        {"io": {}},
        {
            "io.mode": "modbus",
            "io.modbus.enabled": "True",
            "io.indicator_target": "relay",
            "io.modbus.port": "/dev/ttyUSB9",
            "io.modbus.baud": "19200",
            "io.modbus.parity": "E",
            "io.modbus.stopbits": "1",
            "io.modbus.bytesize": "8",
            "io.modbus.timeout_s": "0.6",
            "io.modbus.slave_id": "5",
            "io.relay.slave_id": "6",
            "io.inputs.trigger_capture.enabled": "True",
            "io.inputs.trigger_capture.channel": "2",
            "io.inputs.trigger_capture.active_high": "False",
            "io.inputs.trigger_capture.debounce_ms": "45",
            "io.inputs.part_in_nest.enabled": "True",
            "io.inputs.part_in_nest.channel": "3",
            "io.inputs.part_in_nest.active_high": "False",
            "io.outputs.inspection_pass.channel": "4",
            "io.outputs.inspection_pass.pulse_ms": "1500",
            "io.outputs.inspection_fail.channel": "off",
            "io.outputs.inspection_fail.pulse_ms": "1600",
            "io.outputs.review_needed.channel": "5",
            "io.outputs.review_needed.pulse_ms": "1700",
            "io.outputs.system_ready.channel": "off",
            "io.outputs.system_ready.pulse_ms": "1800",
        },
    )

    assert updated["io"]["mode"] == "modbus"
    assert updated["io"]["modbus"]["port"] == "/dev/ttyUSB9"
    assert updated["io"]["modbus"]["slave_id"] == 5
    assert updated["io"]["relay"]["slave_id"] == 6
    assert updated["io"]["trigger"]["enabled"] is True
    assert updated["io"]["trigger"]["channel"] == 2
    assert updated["io"]["inputs"]["trigger_capture"]["active_high"] is False
    assert updated["io"]["inputs"]["part_in_nest"]["channel"] == 3
    assert updated["io"]["outputs"]["inspection_pass"]["channel"] == 4
    assert updated["io"]["outputs"]["inspection_fail"]["channel"] is None
    assert updated["io"]["relay"]["pass_channel"] == 4
    assert updated["io"]["relay"]["fail_channel"] is None
    assert updated["io"]["pass_pulse_ms"] == 1500
    assert updated["io"]["fail_pulse_ms"] == 1600


def test_format_io_setup_summary_reports_mode_and_assignments() -> None:
    summary = format_io_setup_summary(
        {
            "io": {
                "mode": "modbus",
                "indicator_target": "relay",
                "modbus": {"enabled": True, "slave_id": 1},
                "relay": {"slave_id": 2, "pass_channel": 0, "fail_channel": 1},
                "trigger": {"enabled": True, "channel": 0},
                "inputs": {"part_in_nest": {"enabled": True, "channel": 1}},
            }
        }
    )

    assert "RELAY" in summary
    assert "trigger DI0" in summary
    assert "nest DI1" in summary
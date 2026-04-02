from reference_region_utils import build_reference_regions


def test_build_reference_regions_uses_expected_iterations() -> None:
    calls: list[tuple[str, object, int]] = []

    def fake_dilate(mask, iterations):
        calls.append(("dilate", mask, iterations))
        return f"allowed:{iterations}"

    def fake_erode(mask, iterations):
        calls.append(("erode", mask, iterations))
        return f"required:{iterations}"

    reference_mask = "mask"
    inspection_cfg = {
        "allowed_dilate_iterations": 3,
        "required_erode_iterations": 2,
    }

    allowed_mask, required_mask = build_reference_regions(
        reference_mask,
        inspection_cfg,
        fake_dilate,
        fake_erode,
    )

    assert allowed_mask == "allowed:3"
    assert required_mask == "required:2"
    assert calls == [
        ("dilate", "mask", 3),
        ("erode", "mask", 2),
    ]


def test_build_reference_regions_uses_defaults() -> None:
    calls: list[tuple[str, object, int]] = []

    def fake_dilate(mask, iterations):
        calls.append(("dilate", mask, iterations))
        return "allowed-default"

    def fake_erode(mask, iterations):
        calls.append(("erode", mask, iterations))
        return "required-default"

    allowed_mask, required_mask = build_reference_regions(
        "mask",
        {},
        fake_dilate,
        fake_erode,
    )

    assert allowed_mask == "allowed-default"
    assert required_mask == "required-default"
    assert calls == [
        ("dilate", "mask", 2),
        ("erode", "mask", 1),
    ]
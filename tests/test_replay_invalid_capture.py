from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import replay_inspection


class FakeCv2:
    IMREAD_COLOR = 1

    def __init__(self, image):
        self._image = image

    def imread(self, _path: str, _mode: int):
        return self._image


def test_classify_invalid_capture_reports_missing_image(monkeypatch, tmp_path: Path) -> None:
    fake_cv2 = FakeCv2(image=None)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    missing = tmp_path / "missing.jpg"
    reason = replay_inspection.classify_invalid_capture({"inspection": {"roi": {}}}, missing)

    assert "does not exist" in reason


def test_classify_invalid_capture_reports_bad_roi(monkeypatch, tmp_path: Path) -> None:
    fake_image = SimpleNamespace(shape=(100, 200, 3))
    fake_cv2 = FakeCv2(image=fake_image)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(replay_inspection, "REFERENCE_MASK", tmp_path / "golden_reference_mask.png")
    replay_inspection.REFERENCE_MASK.write_text("placeholder", encoding="utf-8")

    image_path = tmp_path / "sample.jpg"
    image_path.write_text("placeholder", encoding="utf-8")

    reason = replay_inspection.classify_invalid_capture(
        {
            "inspection": {
                "roi": {"x": 180, "y": 10, "width": 40, "height": 20}
            }
        },
        image_path,
    )

    assert reason == "Configured ROI is outside image bounds."


def test_classify_invalid_capture_reports_missing_reference(monkeypatch, tmp_path: Path) -> None:
    fake_image = SimpleNamespace(shape=(100, 200, 3))
    fake_cv2 = FakeCv2(image=fake_image)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(replay_inspection, "REFERENCE_MASK", tmp_path / "missing_reference.png")

    image_path = tmp_path / "sample.jpg"
    image_path.write_text("placeholder", encoding="utf-8")

    reason = replay_inspection.classify_invalid_capture({"inspection": {"roi": {}}}, image_path)

    assert "Reference mask is missing" in reason

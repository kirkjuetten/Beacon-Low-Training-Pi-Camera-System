import numpy as np
import inspection_system.app.registration_engine as registration_engine

from inspection_system.app.registration_engine import register_sample_mask


def test_register_sample_mask_disables_registration_when_alignment_disabled() -> None:
    result = register_sample_mask(
        "sample-mask",
        "reference-mask",
        {"enabled": False, "mode": "moments"},
        object(),
        np,
        lambda *args: (_ for _ in ()).throw(AssertionError("align_sample_mask should not be called")),
    )

    assert result.aligned_mask == "sample-mask"
    assert result.status == "disabled"
    assert result.applied_strategy == "identity"
    assert result.requested_strategy == "moments"
    assert result.transform["angle_deg"] == 0.0
    assert result.observed_anchors == []


def test_register_sample_mask_supports_rigid_refined_runtime() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[5:15, 5:15] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[5:15, 5:15] = 255
    seen_alignment_cfg = {}

    class FakeCv2:
        INTER_NEAREST = "INTER_NEAREST"
        BORDER_CONSTANT = "BORDER_CONSTANT"

        def getRotationMatrix2D(self, center, angle_deg, scale):
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        def warpAffine(self, mask, matrix, size, flags=None, borderMode=None, borderValue=None):
            return mask

        def findNonZero(self, mask):
            points = np.argwhere(mask > 0)
            if points.size == 0:
                return None
            return points[:, ::-1].reshape(-1, 1, 2).astype(np.int32)

        def moments(self, mask, binaryImage=True):
            points = np.argwhere(mask > 0)
            if points.size == 0:
                return {"m00": 0.0, "m10": 0.0, "m01": 0.0}
            return {
                "m00": float(points.shape[0]),
                "m10": float(points[:, 1].sum()),
                "m01": float(points[:, 0].sum()),
            }

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        seen_alignment_cfg.update(alignment_cfg)
        return sample_mask_arg, 0.0, 0, 0

    result = register_sample_mask(
        sample_mask,
        reference_mask,
        {
            "enabled": True,
            "mode": "rigid_refined",
            "registration": {
                "strategy": "rigid_refined",
                "transform_model": "similarity",
                "subpixel_refinement": "template",
            },
        },
        FakeCv2(),
        np,
        fake_align_sample_mask,
    )

    assert seen_alignment_cfg["mode"] == "moments"
    assert result.aligned_mask is sample_mask
    assert result.status == "aligned"
    assert result.applied_strategy == "rigid_refined"
    assert result.runtime_mode == "rigid_refined"
    assert result.requested_strategy == "rigid_refined"
    assert result.transform_model == "similarity"
    assert result.subpixel_refinement == "template"
    assert result.quality["confidence"] > 0.0
    assert result.quality["mean_residual_px"] == 0.0
    assert result.transform["shift_x"] == 0


def test_register_sample_mask_uses_runtime_mode_and_reports_requested_strategy_fallback() -> None:
    sample_mask = np.zeros((20, 20), dtype=np.uint8)
    sample_mask[5:15, 5:15] = 255
    reference_mask = np.zeros((20, 20), dtype=np.uint8)
    reference_mask[5:15, 5:15] = 255

    class FakeCv2:
        def findNonZero(self, mask):
            points = np.argwhere(mask > 0)
            if points.size == 0:
                return None
            return points[:, ::-1].reshape(-1, 1, 2).astype(np.int32)

        def moments(self, mask, binaryImage=True):
            points = np.argwhere(mask > 0)
            if points.size == 0:
                return {"m00": 0.0, "m10": 0.0, "m01": 0.0}
            return {
                "m00": float(points.shape[0]),
                "m10": float(points[:, 1].sum()),
                "m01": float(points[:, 0].sum()),
            }

    def fake_align_sample_mask(sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module):
        assert alignment_cfg["mode"] == "moments"
        return sample_mask_arg, 1.25, 2, -1

    result = register_sample_mask(
        sample_mask,
        reference_mask,
        {
            "enabled": True,
            "mode": "moments",
            "registration": {
                "strategy": "anchor_pair",
                "transform_model": "similarity",
                "anchor_mode": "pair",
                "subpixel_refinement": "phase_correlation",
                "quality_gates": {
                    "min_confidence": 0.9,
                    "max_mean_residual_px": 1.2,
                },
                "datum_frame": {
                    "origin": "anchor_primary",
                    "orientation": "anchor_pair",
                },
            },
        },
        FakeCv2(),
        np,
        fake_align_sample_mask,
    )

    assert result.aligned_mask is sample_mask
    assert result.angle_deg == 1.25
    assert result.shift_x == 2
    assert result.shift_y == -1
    assert result.status == "aligned"
    assert result.applied_strategy == "moments"
    assert result.requested_strategy == "anchor_pair"
    assert result.transform_model == "similarity"
    assert result.anchor_mode == "pair"
    assert result.subpixel_refinement == "phase_correlation"
    assert result.quality_gates == {"min_confidence": 0.9, "max_mean_residual_px": 1.2}
    assert result.datum_frame == {"origin": "anchor_primary", "orientation": "anchor_pair"}
    assert "staged but runtime is using 'moments'" in str(result.fallback_reason)
    assert result.transform == {
        "angle_deg": 1.25,
        "shift_x": 2,
        "shift_y": -1,
        "center": {"x": 10.0, "y": 10.0},
    }


def test_register_sample_mask_supports_anchor_translation_runtime() -> None:
    sample_mask = np.zeros((30, 30), dtype=np.uint8)
    sample_mask[8:10, 7:9] = 255

    def fake_shift_mask(mask, shift_x, shift_y, cv2, np_module):
        assert shift_x == 2
        assert shift_y == 2
        return "shifted-mask"

    original_shift_mask = registration_engine.shift_mask
    registration_engine.shift_mask = fake_shift_mask
    try:
        result = register_sample_mask(
            sample_mask,
            np.zeros((30, 30), dtype=np.uint8),
            {
                "enabled": True,
                "mode": "anchor_translation",
                "max_shift_x": 10,
                "max_shift_y": 10,
                "registration": {
                    "strategy": "anchor_translation",
                    "anchor_mode": "single",
                    "anchors": [
                        {
                            "anchor_id": "anchor_a",
                            "reference_point": {"x": 10, "y": 10},
                            "search_window": {"x": 6, "y": 7, "width": 6, "height": 6},
                        }
                    ],
                },
            },
            object(),
            np,
            lambda *args: (_ for _ in ()).throw(AssertionError("moments aligner should not be called")),
        )
    finally:
        registration_engine.shift_mask = original_shift_mask

    assert result.aligned_mask == "shifted-mask"
    assert result.status == "aligned"
    assert result.applied_strategy == "anchor_translation"
    assert result.shift_x == 2
    assert result.shift_y == 2
    assert result.quality["confidence"] > 0.0
    assert result.quality["mean_residual_px"] < 1.0
    assert result.transform["shift_x"] == 2
    assert result.observed_anchors[0]["anchor_id"] == "anchor_a"
    assert result.observed_anchors[0]["transformed_point"] == {"x": 9.5, "y": 10.5}


def test_register_sample_mask_rejects_when_registration_quality_gate_fails() -> None:
    sample_mask = np.zeros((30, 30), dtype=np.uint8)
    sample_mask[8:10, 7:9] = 255

    def fake_shift_mask(mask, shift_x, shift_y, cv2, np_module):
        return mask

    original_shift_mask = registration_engine.shift_mask
    registration_engine.shift_mask = fake_shift_mask
    try:
        result = register_sample_mask(
            sample_mask,
            np.zeros((30, 30), dtype=np.uint8),
            {
                "enabled": True,
                "mode": "anchor_translation",
                "max_shift_x": 10,
                "max_shift_y": 10,
                "registration": {
                    "strategy": "anchor_translation",
                    "anchor_mode": "single",
                    "quality_gates": {
                        "min_confidence": 1.1,
                    },
                    "anchors": [
                        {
                            "anchor_id": "anchor_a",
                            "reference_point": {"x": 10, "y": 10},
                            "search_window": {"x": 6, "y": 7, "width": 6, "height": 6},
                        }
                    ],
                },
            },
            object(),
            np,
            lambda *args: (_ for _ in ()).throw(AssertionError("moments aligner should not be called")),
        )
    finally:
        registration_engine.shift_mask = original_shift_mask

    assert result.status == "quality_gate_failed"
    assert result.applied_strategy == "anchor_translation"
    assert result.rejection_reason is not None
    assert result.quality_gate_failures[0]["gate_key"] == "min_confidence"
    assert result.quality_gate_failures[0]["cause_code"] == "registration_failure"


def test_register_sample_mask_supports_anchor_pair_runtime() -> None:
    sample_mask = np.zeros((30, 30), dtype=np.uint8)
    sample_mask[20, 10] = 255
    sample_mask[10, 10] = 255
    calls = []

    def fake_rotate_mask(mask, angle_deg, cv2):
        calls.append(("rotate", angle_deg))
        return "rotated-mask"

    def fake_shift_mask(mask, shift_x, shift_y, cv2, np_module):
        calls.append(("shift", shift_x, shift_y))
        return "aligned-mask"

    original_rotate_mask = registration_engine.rotate_mask
    original_shift_mask = registration_engine.shift_mask
    registration_engine.rotate_mask = fake_rotate_mask
    registration_engine.shift_mask = fake_shift_mask
    try:
        result = register_sample_mask(
            sample_mask,
            np.zeros((30, 30), dtype=np.uint8),
            {
                "enabled": True,
                "mode": "anchor_pair",
                "max_angle_deg": 90,
                "max_shift_x": 10,
                "max_shift_y": 10,
                "registration": {
                    "strategy": "anchor_pair",
                    "anchor_mode": "pair",
                    "transform_model": "similarity",
                    "anchors": [
                        {
                            "anchor_id": "primary",
                            "reference_point": {"x": 10, "y": 10},
                            "search_window": {"x": 8, "y": 18, "width": 5, "height": 5},
                        },
                        {
                            "anchor_id": "secondary",
                            "reference_point": {"x": 20, "y": 10},
                            "search_window": {"x": 8, "y": 8, "width": 5, "height": 5},
                        },
                    ],
                },
            },
            object(),
            np,
            lambda *args: (_ for _ in ()).throw(AssertionError("moments aligner should not be called")),
        )
    finally:
        registration_engine.rotate_mask = original_rotate_mask
        registration_engine.shift_mask = original_shift_mask

    assert result.aligned_mask == "rotated-mask"
    assert result.status == "aligned"
    assert result.applied_strategy == "anchor_pair"
    assert result.angle_deg == 90.0
    assert result.shift_x == 0
    assert result.shift_y == 0
    assert result.transform_model == "similarity"
    assert result.quality["confidence"] > 0.0
    assert result.quality["mean_residual_px"] == 0.0
    assert calls == [("rotate", 90.0)]
    assert result.transform == {
        "angle_deg": 90.0,
        "shift_x": 0,
        "shift_y": 0,
        "center": {"x": 15.0, "y": 15.0},
    }
    assert len(result.observed_anchors) == 2
    assert result.observed_anchors[0]["transformed_point"] == {"x": 10.0, "y": 10.0}
    assert result.observed_anchors[1]["transformed_point"] == {"x": 20.0, "y": 10.0}


def test_register_sample_mask_uses_template_matching_when_registration_images_are_available() -> None:
    sample_mask = np.zeros((32, 32), dtype=np.uint8)
    sample_mask[13:18, 9:14] = 255
    reference_mask = np.zeros((32, 32), dtype=np.uint8)
    reference_mask[11:16, 11:16] = 255

    sample_registration_image = np.zeros((32, 32), dtype=np.uint8)
    sample_registration_image[13:18, 9:14] = np.array(
        [
            [10, 40, 70, 40, 10],
            [20, 90, 150, 90, 20],
            [30, 130, 220, 130, 30],
            [20, 90, 150, 90, 20],
            [10, 40, 70, 40, 10],
        ],
        dtype=np.uint8,
    )
    reference_registration_image = np.zeros((32, 32), dtype=np.uint8)
    reference_registration_image[11:16, 11:16] = np.array(
        [
            [10, 40, 70, 40, 10],
            [20, 90, 150, 90, 20],
            [30, 130, 220, 130, 30],
            [20, 90, 150, 90, 20],
            [10, 40, 70, 40, 10],
        ],
        dtype=np.uint8,
    )

    def fake_shift_mask(mask, shift_x, shift_y, cv2, np_module):
        assert shift_x == 2
        assert shift_y == -2
        return "shifted-mask"

    original_shift_mask = registration_engine.shift_mask
    registration_engine.shift_mask = fake_shift_mask
    try:
        result = register_sample_mask(
            sample_mask,
            reference_mask,
            {
                "enabled": True,
                "mode": "anchor_translation",
                "max_shift_x": 10,
                "max_shift_y": 10,
                "registration": {
                    "strategy": "anchor_translation",
                    "anchor_mode": "single",
                    "anchors": [
                        {
                            "anchor_id": "anchor_a",
                            "reference_point": {"x": 13, "y": 13},
                            "search_window": {"x": 7, "y": 9, "width": 10, "height": 10},
                        }
                    ],
                },
            },
            object(),
            np,
            lambda *args: (_ for _ in ()).throw(AssertionError("moments aligner should not be called")),
            sample_registration_image=sample_registration_image,
            reference_registration_image=reference_registration_image,
        )
    finally:
        registration_engine.shift_mask = original_shift_mask

    assert result.status == "aligned"
    assert result.aligned_mask == "shifted-mask"
    assert result.shift_x == 2
    assert result.shift_y == -2
    assert result.quality["confidence"] > 0.7
    assert result.observed_anchors[0]["localization_method"] == "template_match"
    assert result.observed_anchors[0]["image_score"] is not None
    assert result.observed_anchors[0]["mask_score"] is not None


def test_register_sample_mask_refines_moments_runtime_with_registration_images() -> None:
    sample_mask = np.zeros((16, 16), dtype=np.uint8)
    sample_mask[5:9, 4:8] = 255
    reference_mask = np.zeros((16, 16), dtype=np.uint8)
    reference_mask[5:9, 5:9] = 255

    sample_registration_image = np.zeros((16, 16), dtype=np.uint8)
    sample_registration_image[5:9, 4:8] = np.array(
        [
            [10, 60, 40, 10],
            [20, 150, 110, 20],
            [10, 120, 90, 10],
            [5, 30, 20, 5],
        ],
        dtype=np.uint8,
    )
    reference_registration_image = np.zeros((16, 16), dtype=np.uint8)
    reference_registration_image[5:9, 5:9] = np.array(
        [
            [10, 60, 40, 10],
            [20, 150, 110, 20],
            [10, 120, 90, 10],
            [5, 30, 20, 5],
        ],
        dtype=np.uint8,
    )

    class TranslationCv2:
        INTER_NEAREST = 0
        INTER_LINEAR = 1
        BORDER_CONSTANT = 0

        def getRotationMatrix2D(self, center, angle_deg, scale):
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        def warpAffine(self, image, matrix, size, flags=None, borderMode=None, borderValue=None):
            shift_x = int(round(float(matrix[0, 2])))
            shift_y = int(round(float(matrix[1, 2])))
            output = np.zeros_like(image)
            src_y0 = max(0, -shift_y)
            src_y1 = image.shape[0] - max(0, shift_y)
            src_x0 = max(0, -shift_x)
            src_x1 = image.shape[1] - max(0, shift_x)
            dst_y0 = max(0, shift_y)
            dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
            dst_x0 = max(0, shift_x)
            dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
            if src_y1 > src_y0 and src_x1 > src_x0:
                output[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
            return output

        def findNonZero(self, mask):
            points = np.argwhere(mask > 0)
            if points.size == 0:
                return None
            return points[:, ::-1].reshape(-1, 1, 2).astype(np.int32)

        def moments(self, mask, binaryImage=True):
            points = np.argwhere(mask > 0)
            if points.size == 0:
                return {"m00": 0.0, "m10": 0.0, "m01": 0.0}
            return {
                "m00": float(points.shape[0]),
                "m10": float(points[:, 1].sum()),
                "m01": float(points[:, 0].sum()),
            }

    result = register_sample_mask(
        sample_mask,
        reference_mask,
        {
            "enabled": True,
            "mode": "moments",
            "max_shift_x": 2,
            "max_shift_y": 2,
        },
        TranslationCv2(),
        np,
        lambda sample_mask_arg, reference_mask_arg, alignment_cfg, cv2, np_module: (sample_mask_arg, 0.0, 0, 0),
        sample_registration_image=sample_registration_image,
        reference_registration_image=reference_registration_image,
    )

    assert result.status == "aligned"
    assert result.applied_strategy == "moments"
    assert result.shift_x == 1
    assert result.shift_y == 0
    assert result.quality["confidence"] > 0.9
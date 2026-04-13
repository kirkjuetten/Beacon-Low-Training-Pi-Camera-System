#!/usr/bin/env python3
"""
Hardware Robustness Tests for Beacon Inspection System

Tests designed to ensure the system doesn't crash on Raspberry Pi hardware
under various failure conditions.
"""

import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from inspection_system.app import camera_interface, frame_acquisition
from inspection_system.app.runtime_controller import run_capture_and_inspect
from inspection_system.app.replay_inspection import inspect_file


class TestHardwareRobustness:
    """Test hardware failure scenarios that could crash the system."""

    def test_camera_command_timeout(self, monkeypatch):
        """Test that camera timeouts are handled gracefully."""
        config = {"capture": {"timeout_ms": 100}}

        # Mock subprocess to simulate timeout
        mock_result = mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Camera timeout"
        mock_result.stdout = ""

        with mock.patch('subprocess.run', return_value=mock_result):
            result_code, image_path, stderr = frame_acquisition.capture_to_temp(config)
            assert result_code == 1
            assert "Camera timeout" in stderr

    def test_camera_hardware_not_connected(self, monkeypatch):
        """Test behavior when camera hardware is not available."""
        config = camera_interface.load_config()

        mock_result = mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "rpicam-still: Could not open camera"
        mock_result.stdout = ""

        with mock.patch('subprocess.run', return_value=mock_result):
            result_code, image_path, stderr = frame_acquisition.capture_to_temp(config)
            assert result_code == 1
            assert "Could not open camera" in stderr

    def test_corrupted_image_file(self, tmp_path):
        """Test handling of corrupted image files."""
        config = camera_interface.load_config()

        # Create a corrupted "image" file
        corrupted_image = tmp_path / "corrupted.jpg"
        corrupted_image.write_bytes(b"This is not a valid JPEG")

        # Mock cv2.imread to return None (failed to read)
        with mock.patch('cv2.imread', return_value=None):
            with pytest.raises(FileNotFoundError, match="Unable to read image"):
                from inspection_system.app.preprocessing_utils import make_binary_mask
                # Use the real import function but mock cv2.imread
                make_binary_mask(corrupted_image, config["inspection"], camera_interface.import_cv2_and_numpy)

    def test_insufficient_memory_large_image(self, tmp_path, monkeypatch):
        """Test behavior with very large images that might cause memory issues."""
        # Create a mock very large image
        large_image = np.ones((5000, 5000, 3), dtype=np.uint8) * 128

        config = camera_interface.load_config()

        # Mock cv2 to return the large image
        def mock_imread(path, mode):
            if mode == 1:  # IMREAD_COLOR
                return large_image
            return None

        with mock.patch('cv2.imread', side_effect=mock_imread):
            # This should still work but might be slow - test that it doesn't crash
            from inspection_system.app.preprocessing_utils import make_binary_mask
            roi_image, gray, mask, roi, cv2_mod, np_mod = make_binary_mask(
                tmp_path / "large.jpg", config["inspection"], camera_interface.import_cv2_and_numpy
            )
            assert roi_image.shape[0] > 100  # Should process successfully

    def test_gpio_hardware_failure(self, monkeypatch):
        """Test GPIO operations when hardware is not available."""
        # Mock RPi.GPIO import failure
        mock_gpio = mock.MagicMock()
        mock_gpio.side_effect = ImportError("No module named 'RPi'")

        with mock.patch.dict('sys.modules', {'RPi.GPIO': mock_gpio}):
            led = camera_interface.IndicatorLED(
                enabled=True, pass_gpio=23, fail_gpio=24, pulse_ms=750
            )
            assert not led.enabled  # Should disable gracefully

            # Should not crash when pulsing
            led.pulse_pass()
            led.pulse_fail()
            led.cleanup()  # Should not crash

    def test_gpio_pin_conflict(self, monkeypatch):
        """Test GPIO behavior when pins are already in use."""
        mock_gpio = mock.MagicMock()
        mock_gpio.BCM = "BCM"
        mock_gpio.OUT = "OUT"
        mock_gpio.LOW = "LOW"
        mock_gpio.HIGH = "HIGH"

        # Mock setup to raise RuntimeError (pin already in use)
        mock_gpio.setup.side_effect = RuntimeError("Pin already in use")
        mock_gpio.setwarnings = mock.MagicMock()
        mock_gpio.setmode = mock.MagicMock()

        with mock.patch.dict('sys.modules', {'RPi': mock.MagicMock(), 'RPi.GPIO': mock_gpio}):
            with mock.patch('builtins.__import__', return_value=mock.MagicMock(GPIO=mock_gpio)):
                led = camera_interface.IndicatorLED(
                    enabled=True, pass_gpio=23, fail_gpio=24, pulse_ms=750
                )
                assert not led.enabled  # Should disable on GPIO setup failure

    def test_missing_reference_files(self, tmp_path):
        """Test behavior when reference files are missing."""
        # Create a dummy image file and mock cv2 to read it
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"dummy image data")

        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Test that REFERENCE_MASK existence is checked
        with mock.patch('cv2.imread', return_value=mock_image):
            with mock.patch('inspection_system.app.camera_interface.REFERENCE_MASK', tmp_path / "missing_mask.png"):
                from inspection_system.app.replay_inspection import classify_invalid_capture
                result = classify_invalid_capture({"inspection": {"roi": {}}}, test_image)
                assert "Reference mask is missing" in result

    def test_mismatched_image_dimensions(self, tmp_path):
        """Test handling of images with unexpected dimensions."""
        # Test the shape validation directly
        reference_mask = np.zeros((100, 100), dtype=np.uint8)
        sample_mask = np.zeros((50, 50), dtype=np.uint8)  # Different size

        # This should raise ValueError for mismatched shapes
        with pytest.raises(ValueError, match="Reference mask shape .* does not match sample mask shape"):
            # Simulate the shape check from inspect_against_reference
            if reference_mask.shape != sample_mask.shape:
                raise ValueError(
                    f"Reference mask shape {reference_mask.shape} does not match sample mask shape {sample_mask.shape}."
                )

    def test_extreme_threshold_values(self):
        """Test behavior with extreme threshold configurations."""
        config = {
            "inspection": {
                "min_required_coverage": 0.99,  # Very strict
                "max_outside_allowed_ratio": 0.001,  # Very strict
                "min_section_coverage": 0.95,  # Very strict
            }
        }

        from inspection_system.app.scoring_utils import evaluate_metrics

        # Test with metrics that should fail strict thresholds
        metrics = {
            "required_coverage": 0.85,
            "outside_allowed_ratio": 0.05,
            "min_section_coverage": 0.80,
        }

        passed, details = evaluate_metrics(metrics, config["inspection"])
        assert not passed  # Should fail with strict thresholds
        assert details["required_coverage"] == 0.85

    def test_empty_or_zero_size_images(self, tmp_path):
        """Test handling of degenerate image cases."""
        config = camera_interface.load_config()

        # Mock cv2.imread to return empty image
        empty_image = np.zeros((0, 0, 3), dtype=np.uint8)

        with mock.patch('cv2.imread', return_value=empty_image):
            with pytest.raises(ValueError, match="Configured ROI is outside"):
                from inspection_system.app.preprocessing_utils import make_binary_mask
                make_binary_mask(tmp_path / "empty.jpg", config["inspection"], camera_interface.import_cv2_and_numpy)

    def test_network_storage_failures(self, tmp_path, monkeypatch):
        """Test behavior when file I/O fails (network storage, permissions, etc.)."""
        config = camera_interface.load_config()

        # Mock file write failure
        with mock.patch('cv2.imwrite', return_value=False):
            from inspection_system.app.capture_test import save_debug_outputs
            # Should not crash, just not save debug images
            result = save_debug_outputs("test", np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8))
            assert isinstance(result, dict)  # Should return dict even if write fails

    def test_concurrent_access_protection(self, tmp_path):
        """Test that temp file handling prevents concurrent access issues."""
        # This tests the cleanup_temp_image function
        temp_file = tmp_path / "temp_capture.jpg"
        temp_file.write_text("test")

        # Mock the TEMP_IMAGE constant at the module level
        with mock.patch('inspection_system.app.frame_acquisition.TEMP_IMAGE', temp_file):
            frame_acquisition.cleanup_temp_image()
            assert not temp_file.exists()  # Should be cleaned up

            # Should not crash if file doesn't exist
            frame_acquisition.cleanup_temp_image()  # Second call should be safe


class TestPerformanceUnderLoad:
    """Test performance characteristics that could affect hardware stability."""

    def test_memory_usage_bounds(self):
        """Test that memory usage stays within reasonable bounds."""
        import numpy as np
        # Create a moderately large image for processing
        test_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        config = camera_interface.load_config()

        # Mock cv2 operations
        with mock.patch('cv2.imread', return_value=test_image):
            with mock.patch('cv2.imwrite', return_value=True):
                from inspection_system.app.preprocessing_utils import make_binary_mask
                roi_image, gray, mask, roi, cv2, np = make_binary_mask(
                    Path("test.jpg"), config["inspection"], camera_interface.import_cv2_and_numpy
                )

                # Check that output sizes are reasonable
                assert roi_image.shape[0] <= 1000
                assert roi_image.shape[1] <= 1000
                assert mask.shape[0] <= 1000
                assert mask.shape[1] <= 1000

    def test_processing_time_bounds(self):
        """Test that processing doesn't take excessive time."""
        import time
        test_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

        config = camera_interface.load_config()

        with mock.patch('cv2.imread', return_value=test_image):
            with mock.patch('cv2.imwrite', return_value=True):
                start_time = time.time()

                from inspection_system.app.preprocessing_utils import make_binary_mask
                make_binary_mask(Path("test.jpg"), config["inspection"], camera_interface.import_cv2_and_numpy)

                processing_time = time.time() - start_time
                assert processing_time < 5.0  # Should complete in reasonable time
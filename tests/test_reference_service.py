from inspection_system.app.reference_service import check_reference_settings_match


def test_check_reference_settings_match_reports_context_mismatches(monkeypatch) -> None:
    monkeypatch.setattr(
        'inspection_system.app.reference_service.load_reference_metadata',
        lambda: {
            'roi': {'x': 10, 'y': 20, 'width': 30, 'height': 40},
            'threshold': {'type': 'otsu', 'value': 180.0},
            'morphology': {'reference_erode_iterations': 1, 'reference_dilate_iterations': 1},
            'inspection_context': {
                'inspection_mode': 'mask_only',
                'reference_strategy': 'golden_only',
                'blend_mode': 'hard_only',
                'tolerance_mode': 'balanced',
            },
            'alignment': {'tolerance_profile': 'balanced'},
        },
    )

    matched, warning = check_reference_settings_match(
        {
            'inspection': {
                'roi': {'x': 11, 'y': 20, 'width': 30, 'height': 40},
                'threshold_mode': 'fixed',
                'threshold_value': 170,
                'reference_erode_iterations': 2,
                'reference_dilate_iterations': 3,
                'inspection_mode': 'full',
                'reference_strategy': 'hybrid',
                'blend_mode': 'blend_balanced',
                'tolerance_mode': 'forgiving',
            },
            'alignment': {'tolerance_profile': 'strict'},
        }
    )

    assert matched is False
    assert 'ROI mismatch' in warning
    assert 'Threshold type mismatch' in warning
    assert 'Reference erode iterations mismatch' in warning
    assert 'inspection mode mismatch' in warning
    assert 'reference strategy mismatch' in warning
    assert 'blend mode mismatch' in warning
    assert 'tolerance mode mismatch' in warning
    assert 'alignment tolerance profile mismatch' in warning
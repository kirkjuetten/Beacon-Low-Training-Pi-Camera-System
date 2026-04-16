from inspection_system.app.reference_service import check_reference_settings_match, list_runtime_reference_candidates


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


def test_list_runtime_reference_candidates_hybrid_includes_golden_and_active_variant(tmp_path) -> None:
    reference_dir = tmp_path / 'reference'
    reference_dir.mkdir(parents=True, exist_ok=True)
    golden_mask = reference_dir / 'golden_reference_mask.png'
    golden_image = reference_dir / 'golden_reference_image.png'
    golden_mask.write_bytes(b'mask')
    golden_image.write_bytes(b'image')
    (reference_dir / 'ref_meta.json').write_text(
        '{\n  "reference_asset": {"reference_id": "golden", "label": "Golden Reference"}\n}\n',
        encoding='utf-8',
    )

    variant_dir = reference_dir / 'reference_variants' / 'active' / 'good_ref_1'
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / 'reference_mask.png').write_bytes(b'mask')
    (variant_dir / 'reference_image.png').write_bytes(b'image')
    (variant_dir / 'ref_meta.json').write_text(
        '{\n  "reference_asset": {"reference_id": "good_ref_1", "label": "Approved Good 1", "role": "candidate"}\n}\n',
        encoding='utf-8',
    )

    candidates = list_runtime_reference_candidates(
        {'inspection': {'reference_strategy': 'hybrid'}},
        {
            'reference_dir': reference_dir,
            'reference_mask': golden_mask,
            'reference_image': golden_image,
        },
    )

    assert [candidate['reference_id'] for candidate in candidates] == ['golden', 'good_ref_1']


def test_list_runtime_reference_candidates_multi_good_prefers_active_variants(tmp_path) -> None:
    reference_dir = tmp_path / 'reference'
    reference_dir.mkdir(parents=True, exist_ok=True)
    golden_mask = reference_dir / 'golden_reference_mask.png'
    golden_image = reference_dir / 'golden_reference_image.png'
    golden_mask.write_bytes(b'mask')
    golden_image.write_bytes(b'image')

    variant_dir = reference_dir / 'reference_variants' / 'active' / 'good_ref_2'
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / 'reference_mask.png').write_bytes(b'mask')
    (variant_dir / 'reference_image.png').write_bytes(b'image')
    (variant_dir / 'ref_meta.json').write_text(
        '{\n  "reference_asset": {"reference_id": "good_ref_2", "label": "Approved Good 2", "role": "candidate"}\n}\n',
        encoding='utf-8',
    )

    candidates = list_runtime_reference_candidates(
        {'inspection': {'reference_strategy': 'multi_good_experimental'}},
        {
            'reference_dir': reference_dir,
            'reference_mask': golden_mask,
            'reference_image': golden_image,
        },
    )

    assert [candidate['reference_id'] for candidate in candidates] == ['good_ref_2']
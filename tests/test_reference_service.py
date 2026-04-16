import json

import inspection_system.app.reference_service as reference_service
from inspection_system.app.reference_service import (
    check_reference_settings_match,
    list_runtime_reference_candidates,
    list_anomaly_training_samples,
    train_anomaly_model_from_samples,
)


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


def test_list_anomaly_training_samples_reads_active_entries(tmp_path) -> None:
    reference_dir = tmp_path / 'reference'
    sample_dir = reference_dir / 'anomaly_good_library' / 'active' / 'sample_a'
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / 'sample_meta.json').write_text(
        json.dumps(
            {
                'sample_asset': {'sample_id': 'sample_a', 'label': 'Approved Good A'},
                'features': [0.1, 0.2, 0.3],
            }
        ),
        encoding='utf-8',
    )

    entries = list_anomaly_training_samples({'reference_dir': reference_dir}, states=('active',))

    assert len(entries) == 1
    assert entries[0]['sample_id'] == 'sample_a'
    assert entries[0]['features'] == [0.1, 0.2, 0.3]


def test_train_anomaly_model_from_samples_requires_minimum_good_samples(tmp_path) -> None:
    reference_dir = tmp_path / 'reference'
    active_dir = reference_dir / 'anomaly_good_library' / 'active'
    for index in range(2):
        sample_dir = active_dir / f'sample_{index}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / 'sample_meta.json').write_text(
            json.dumps({'sample_asset': {'sample_id': f'sample_{index}'}, 'features': [0.1, 0.2, 0.3]}),
            encoding='utf-8',
        )

    result = train_anomaly_model_from_samples({'inspection': {}}, {'reference_dir': reference_dir}, minimum_samples=3)

    assert result['rebuilt'] is False
    assert 'Need at least 3 approved-good samples' in result['reason']
    assert not (reference_dir / 'anomaly_model.pkl').exists()


def test_train_anomaly_model_from_samples_persists_model_and_metadata(tmp_path, monkeypatch) -> None:
    reference_dir = tmp_path / 'reference'
    active_dir = reference_dir / 'anomaly_good_library' / 'active'
    for index in range(3):
        sample_dir = active_dir / f'sample_{index}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / 'sample_meta.json').write_text(
            json.dumps({'sample_asset': {'sample_id': f'sample_{index}'}, 'features': [0.1, 0.2, 0.3]}),
            encoding='utf-8',
        )

    class FakeDetector:
        def __init__(self, model_path):
            self.model_path = model_path
            self.trained_rows = None

        def train(self, features_list):
            self.trained_rows = features_list

        def save_model(self):
            self.model_path.write_bytes(b'model')

    monkeypatch.setattr(reference_service, 'AnomalyDetector', FakeDetector)

    result = train_anomaly_model_from_samples(
        {'inspection': {'inspection_mode': 'mask_and_ml'}},
        {'reference_dir': reference_dir},
        minimum_samples=3,
    )

    assert result['rebuilt'] is True
    assert (reference_dir / 'anomaly_model.pkl').exists()
    metadata = json.loads((reference_dir / 'anomaly_model_meta.json').read_text(encoding='utf-8'))
    assert metadata['trained_sample_count'] == 3
from inspection_system.app.inspection_runtime_context import (
    InspectionRuntimeContext,
    build_inspection_runtime_context,
    refresh_inspection_runtime_context,
)


def test_build_inspection_runtime_context_loads_paths_refs_and_detector() -> None:
    context = build_inspection_runtime_context(
        {"inspection": {"inspection_mode": "mask_only"}},
        active_paths_loader=lambda: {"config_file": "cfg.json", "reference_dir": "refs"},
        reference_candidates_loader=lambda config, active_paths: [{"id": "ref_1", "paths": active_paths}],
        anomaly_detector_loader=lambda active_paths: {"detector": active_paths["reference_dir"]},
    )

    assert isinstance(context, InspectionRuntimeContext)
    assert context.active_paths["config_file"] == "cfg.json"
    assert context.reference_candidates[0]["id"] == "ref_1"
    assert context.anomaly_detector == {"detector": "refs"}


def test_refresh_inspection_runtime_context_reloads_state() -> None:
    context = InspectionRuntimeContext(
        config={"inspection": {"inspection_mode": "mask_only"}},
        active_paths={"config_file": "old.json", "reference_dir": "old_refs"},
        reference_candidates=[{"id": "old"}],
        anomaly_detector={"detector": "old"},
    )

    refreshed = refresh_inspection_runtime_context(
        context,
        config={"inspection": {"inspection_mode": "full"}},
        active_paths_loader=lambda: {"config_file": "new.json", "reference_dir": "new_refs"},
        reference_candidates_loader=lambda config, active_paths: [{"id": config["inspection"]["inspection_mode"], "dir": active_paths["reference_dir"]}],
        anomaly_detector_loader=lambda active_paths: {"detector": active_paths["config_file"]},
    )

    assert refreshed is context
    assert context.config["inspection"]["inspection_mode"] == "full"
    assert context.active_paths["config_file"] == "new.json"
    assert context.reference_candidates == [{"id": "full", "dir": "new_refs"}]
    assert context.anomaly_detector == {"detector": "new.json"}
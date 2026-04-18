from inspection_system.app.training_assets import (
    commit_pending_training_records,
    discard_pending_training_records,
    stage_training_assets,
)


def test_stage_training_assets_stages_reference_and_anomaly_for_good_records(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"img")
    record = {
        "record_id": "feedback_1",
        "reference_candidate_id": None,
        "reference_candidate_state": None,
        "anomaly_sample_id": None,
        "anomaly_sample_state": None,
    }
    called = {"reference": None, "sample": None}

    def fake_stage_reference_candidate(config, source_image_path, **kwargs):
        called["reference"] = (config, source_image_path, kwargs)
        return True, {"reference_id": "candidate_1", "state": "pending"}

    def fake_stage_anomaly_training_sample(config, source_image_path, **kwargs):
        called["sample"] = (config, source_image_path, kwargs)
        return True, {"sample_id": "sample_1", "state": "pending"}

    stage_training_assets(
        record,
        {"inspection": {"reference_strategy": "hybrid"}},
        final_class="good",
        image_path=image_path,
        record_label_index=1,
        reference_strategy="hybrid",
        active_paths=None,
        stage_reference_candidate=fake_stage_reference_candidate,
        stage_anomaly_training_sample=fake_stage_anomaly_training_sample,
    )

    assert called["reference"][1] == image_path
    assert called["sample"][1] == image_path
    assert record["reference_candidate_id"] == "candidate_1"
    assert record["anomaly_sample_id"] == "sample_1"


def test_commit_and_discard_pending_training_records_update_states() -> None:
    records = [
        {
            "feedback": "approve",
            "final_class": "good",
            "learning_state": "pending",
            "reference_candidate_id": "candidate_1",
            "reference_candidate_state": "pending",
            "anomaly_sample_id": "sample_1",
            "anomaly_sample_state": "pending",
        }
    ]
    activated = []
    discarded = []

    committed = commit_pending_training_records(
        records,
        active_paths=None,
        resolve_learning_class=lambda record: record.get("final_class"),
        activate_reference_candidate=lambda candidate_id: activated.append(("reference", candidate_id)) or True,
        activate_anomaly_training_sample=lambda sample_id: activated.append(("sample", sample_id)) or True,
    )

    assert committed == 1
    assert activated == [("reference", "candidate_1"), ("sample", "sample_1")]
    assert records[0]["learning_state"] == "committed"
    assert records[0]["reference_candidate_state"] == "active"
    assert records[0]["anomaly_sample_state"] == "active"

    records[0]["learning_state"] = "pending"
    records[0]["reference_candidate_state"] = "pending"
    records[0]["anomaly_sample_state"] = "pending"

    discarded_count = discard_pending_training_records(
        records,
        active_paths=None,
        discard_reference_candidate=lambda candidate_id, state="pending": discarded.append(("reference", candidate_id, state)) or True,
        discard_anomaly_training_sample=lambda sample_id, state="pending": discarded.append(("sample", sample_id, state)) or True,
    )

    assert discarded_count == 1
    assert discarded == [("reference", "candidate_1", "pending"), ("sample", "sample_1", "pending")]
    assert records[0]["learning_state"] == "discarded"
    assert records[0]["reference_candidate_state"] == "discarded"
    assert records[0]["anomaly_sample_state"] == "discarded"
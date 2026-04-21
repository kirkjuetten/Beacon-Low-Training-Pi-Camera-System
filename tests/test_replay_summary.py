from replay_summary import summarize_results


def test_summarize_results_counts_statuses():
    results = [
        {"status": "PASS", "registration_status": "aligned"},
        {"status": "FAIL", "failed_lane_ids": ["geometry"], "failed_authoritative_lane_ids": ["geometry"], "registration_status": "aligned"},
        {"status": "FAIL", "failed_lane_ids": ["print"], "failed_authoritative_lane_ids": ["print"], "registration_status": "quality_gate_failed"},
        {"status": "INVALID_CAPTURE"},
        {"status": "CONFIG_ERROR"},
    ]

    summary = summarize_results(results)

    assert summary["status"] == "SUMMARY"
    assert summary["total_images"] == 5
    assert summary["counts"]["PASS"] == 1
    assert summary["counts"]["FAIL"] == 2
    assert summary["counts"]["INVALID_CAPTURE"] == 1
    assert summary["counts"]["CONFIG_ERROR"] == 1
    assert summary["failed_lane_counts"]["geometry"] == 1
    assert summary["failed_lane_counts"]["print"] == 1
    assert summary["registration_status_counts"]["aligned"] == 2
    assert summary["registration_status_counts"]["quality_gate_failed"] == 1

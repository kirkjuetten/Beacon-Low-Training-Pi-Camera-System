from replay_summary import summarize_results


def test_summarize_results_counts_statuses():
    results = [
        {"status": "PASS"},
        {"status": "FAIL"},
        {"status": "FAIL"},
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

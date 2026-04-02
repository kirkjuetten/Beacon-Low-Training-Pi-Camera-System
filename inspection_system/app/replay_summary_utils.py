from result_status import CONFIG_ERROR, FAIL, INVALID_CAPTURE, PASS, SUMMARY

SUMMARY_ORDER = [PASS, FAIL, INVALID_CAPTURE, CONFIG_ERROR]


def summarize_results(results: list[dict]) -> dict:
    counts = {key: 0 for key in SUMMARY_ORDER}
    for result in results:
        status = result.get("status", CONFIG_ERROR)
        counts[status] = counts.get(status, 0) + 1

    return {
        "status": SUMMARY,
        "total_images": len(results),
        "counts": counts,
    }
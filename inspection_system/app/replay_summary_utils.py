from result_status import CONFIG_ERROR, FAIL, INVALID_CAPTURE, PASS, SUMMARY

SUMMARY_ORDER = [PASS, FAIL, INVALID_CAPTURE, CONFIG_ERROR]


def summarize_results(results: list[dict]) -> dict:
    counts = {key: 0 for key in SUMMARY_ORDER}
    failed_lane_counts: dict[str, int] = {}
    failed_authoritative_lane_counts: dict[str, int] = {}
    registration_status_counts: dict[str, int] = {}
    for result in results:
        status = result.get("status", CONFIG_ERROR)
        counts[status] = counts.get(status, 0) + 1

        registration_status = result.get("registration_status")
        if registration_status:
            registration_key = str(registration_status)
            registration_status_counts[registration_key] = registration_status_counts.get(registration_key, 0) + 1

        for lane_id in result.get("failed_lane_ids", []) or []:
            normalized_lane_id = str(lane_id)
            failed_lane_counts[normalized_lane_id] = failed_lane_counts.get(normalized_lane_id, 0) + 1

        for lane_id in result.get("failed_authoritative_lane_ids", []) or []:
            normalized_lane_id = str(lane_id)
            failed_authoritative_lane_counts[normalized_lane_id] = (
                failed_authoritative_lane_counts.get(normalized_lane_id, 0) + 1
            )

    return {
        "status": SUMMARY,
        "total_images": len(results),
        "counts": counts,
        "failed_lane_counts": failed_lane_counts,
        "failed_authoritative_lane_counts": failed_authoritative_lane_counts,
        "registration_status_counts": registration_status_counts,
    }
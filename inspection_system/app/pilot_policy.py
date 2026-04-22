from __future__ import annotations


CAPTURE_DATASET_SPLITS = ("tuning", "validation", "regression")
CAPTURE_BUCKETS = ("good", "reject", "borderline", "invalid_capture")

DEFAULT_SUPERVISED_PILOT_TARGETS = {
    "tuning": {
        "good": 10,
        "reject": 5,
        "invalid_capture": 2,
    },
    "validation": {
        "good": 5,
        "reject": 3,
        "invalid_capture": 2,
    },
    "regression": {
        "good": 5,
        "reject": 3,
        "invalid_capture": 2,
    },
}

DEFAULT_MANUAL_FLOOR_GATES = [
    "Engineering present at line start with authority to stop the run.",
    "Controlled challenge kit staged at the station and segregated from production parts.",
    "First lot executed as a supervised learning run with challenge inserts at a defined cadence.",
]


def _coerce_nonnegative_int(value, fallback: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(0, coerced)


def resolve_supervised_pilot_policy(config: dict | None) -> dict:
    policy = config.get("pilot_readiness", {}) if isinstance(config, dict) else {}
    target_overrides = policy.get("targets", {}) if isinstance(policy, dict) else {}

    resolved_targets: dict[str, dict[str, int]] = {}
    for split in CAPTURE_DATASET_SPLITS:
        default_split_targets = DEFAULT_SUPERVISED_PILOT_TARGETS.get(split, {})
        configured_split_targets = target_overrides.get(split, {}) if isinstance(target_overrides, dict) else {}
        resolved_targets[split] = {}
        for bucket, default_target in default_split_targets.items():
            resolved_targets[split][bucket] = _coerce_nonnegative_int(
                configured_split_targets.get(bucket, default_target),
                default_target,
            )

    manual_floor_gates = policy.get("manual_floor_gates") if isinstance(policy, dict) else None
    if isinstance(manual_floor_gates, list):
        resolved_manual_floor_gates = [
            str(gate).strip() for gate in manual_floor_gates if str(gate).strip()
        ]
        if not resolved_manual_floor_gates:
            resolved_manual_floor_gates = list(DEFAULT_MANUAL_FLOOR_GATES)
    else:
        resolved_manual_floor_gates = list(DEFAULT_MANUAL_FLOOR_GATES)

    return {
        "targets": resolved_targets,
        "manual_floor_gates": resolved_manual_floor_gates,
    }
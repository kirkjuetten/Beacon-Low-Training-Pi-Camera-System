# Supervised Pilot Runbook

This runbook is the staged and phased path to the first supervised floor pilot.

It is intentionally narrow:

- engineering present
- controlled challenge parts on hand
- learning run first
- supervised launch second

## Branch

Use one branch for pilot-readiness execution and keep Pi pulls explicit.

Recommended branch for this work:

```text
feature/supervised-pilot-readiness
```

## Stage 0 - Clean Branch And Pi Sync

Goal: keep local and Pi state unambiguous.

1. Work only from a dedicated feature branch.
2. Keep the repo clean before each Pi pull.
3. Update the Pi with the branch explicitly:

```bash
./scripts/pi-update-branch.sh feature/supervised-pilot-readiness
```

4. On the Pi, run:

```bash
python3 -m inspection_system.app.capture_test pilot-readiness
python3 -m inspection_system.app.capture_test quick-check
```

Exit criteria:

- Pi is on the same branch as engineering work.
- No local drift or mystery branch state.
- Quick functional smoke path still runs.

## Stage 1 - Recipe Isolation

Goal: each pilot recipe lives in its own project.

1. Create or switch to a dedicated project for the light-pipe recipe.
2. Create or switch to a separate dedicated project for the white-pad-print recipe.
3. Never share references, training data, or thresholds across those two recipes.

Exit criteria:

- The active project is correct for the recipe under test.
- Reference assets, logs, and test-data folders are project-scoped.

## Stage 2 - Commission Baseline

Goal: close the registration and reference setup before floor work.

1. Capture the golden reference for the active recipe.
2. Confirm registration baseline is stamped for the current alignment settings.
3. If the recipe uses multiple approved-good references, activate the required variants.
4. Collect enough approved-good parts to satisfy the commissioning target.
5. If ML-backed mode is selected, rebuild the anomaly model before pilot use.

Exit criteria:

- `pilot-readiness` reports commissioning ready.
- No pending approved-good updates remain.

## Stage 3 - Controlled Challenge Set

Goal: prove the recipe against saved, labeled evidence before line launch.

Minimum supervised pilot target:

- `tuning`: 10 good, 5 reject, 2 invalid-capture
- `validation`: 5 good, 3 reject, 2 invalid-capture
- `regression`: 5 good, 3 reject, 2 invalid-capture

Recommended challenge categories:

- good parts with normal process variation
- clear reject parts for the true first target defect
- invalid captures: no part, wrong part, clipped frame, blur, glare, bad seat

Recipe-specific first target defects:

- light pipes: missing feature, gross shift, obvious malformed geometry
- white pad print on black part: missing print, gross shift, gross incomplete transfer, obvious smear

Exit criteria:

- Validation mismatches are zero.
- Regression mismatches are zero.
- Any remaining tuning mismatches are reviewed and documented.

## Stage 4 - Learning Run

Goal: close the supervised learning loop with engineering present.

1. Run training against the active project.
2. Use the controlled challenge parts during the learning pass.
3. Press Update and commit the resulting approved-good learning state.
4. Re-run `pilot-readiness`.

Exit criteria:

- Pending approved-good records are zero.
- The learning run is closed and the current config is the config you intend to launch.

## Stage 5 - Supervised Floor Pilot

Goal: launch the first line trial with the correct safeguards.

Required floor conditions:

1. Engineering is physically present at line start.
2. Controlled challenge kit is staged and segregated from production parts.
3. The first lot is treated as a supervised run, not unattended production.

Execution order:

1. Run `pilot-readiness` one final time.
2. Launch production with the active recipe.
3. Insert controlled challenge parts at a defined cadence.
4. Record any false accept, false reject, or invalid-capture miss immediately.
5. Stop and retune if the pilot escapes the agreed defect boundary.

Exit criteria:

- The pilot stays within the agreed defect scope.
- Engineering agrees the recipe is stable enough for the next tightening step.

## Notes On Scope

This runbook is for the first supervised pilot, not the final autonomous production release.

For the current repo, that means:

- light pipes are expected to reach this stage first
- white pad print is expected to enter this stage only after the saved-image dataset exists
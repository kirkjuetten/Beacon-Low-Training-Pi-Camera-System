# Config Tuning Guide

This guide explains each setting shown on the Config + Preview page.

## Capture Settings

### Capture Timeout (ms) (`capture.timeout_ms`)
- Purpose: How long the camera command waits before returning.
- Increase when: Captures fail on slow startup or auto-adjust delays.
- Decrease when: You want fast failure if hardware is unavailable.

### Shutter (us) (`capture.shutter_us`)
- Purpose: Exposure time in microseconds.
- Increase when: Image is too dark.
- Decrease when: Image is too bright or motion blur appears.

## Inspection Mode and Gates

### Inspection Mode (`inspection.inspection_mode`)
- `mask_only`: Uses mask-based coverage checks only.
- `mask_and_ssim`: Adds SSIM and MSE gates.
- `mask_and_ml`: Adds anomaly score gate.
- `full`: Enables all gates.

Choose `mask_only` for baseline setup and move to richer modes after baseline is stable.

## Mask / Coverage Thresholds

### Threshold Mode (`inspection.threshold_mode`)
- Purpose: Chooses how the binary mask threshold is computed.
- `fixed`: Uses `threshold_value` with normal polarity.
- `fixed_inv`: Uses `threshold_value` with inverted polarity.
- `otsu`: Auto-picks threshold from image histogram with normal polarity.
- `otsu_inv`: Auto-picks threshold with inverted polarity.

Start with `otsu` if lighting is variable. Use `fixed` or `fixed_inv` when lighting is controlled and you want repeatable mask behavior.

### Threshold Value (`inspection.threshold_value`)
- Purpose: Fixed threshold level for mask generation.
- Increase when: Too much background/noise is being marked.
- Decrease when: Real feature regions are being missed.

### Min Feature Pixels (`inspection.min_feature_pixels`)
- Purpose: Minimum detected white pixels required to accept a reference capture.
- Increase when: You want to reject weak/empty references.
- Decrease when: Valid small-feature references are being rejected.

### Min Required Coverage (`inspection.min_required_coverage`)
- Purpose: Minimum fraction of required region that must be covered.
- Higher value: Stricter pass criteria.
- Lower value: More forgiving.

### Max Outside Allowed (`inspection.max_outside_allowed_ratio`)
- Purpose: Maximum fraction of sample mask outside allowed region.
- Lower value: Stricter about extra marks/noise.
- Higher value: More tolerance for spill/noise.

### Min Section Coverage (`inspection.min_section_coverage`)
- Purpose: Minimum per-section coverage floor across section columns.
- Higher value: Stricter on local misses.
- Lower value: More tolerance for local weak regions.

## Optional Quality Gates

These are active only if the selected mode includes them and a value is set.

### Min SSIM (optional) (`inspection.min_ssim`)
- Purpose: Minimum structural similarity score.
- Higher value: Stricter similarity requirement.

### Max MSE (optional) (`inspection.max_mse`)
- Purpose: Maximum mean squared error.
- Lower value: Stricter difference limit.

### Min Anomaly Score (optional) (`inspection.min_anomaly_score`)
- Purpose: Minimum anomaly model score.
- Higher value: Stricter anomaly acceptance.

## Workflow / Diagnostics

### Save Debug Images (`inspection.save_debug_images`)
- Purpose: Save debug output images for troubleshooting.
- `true`: Better troubleshooting, more file output.
- `false`: Less disk churn, fewer artifacts.

### Alignment Enabled (`alignment.enabled`)
- Purpose: Align sample mask to reference before scoring.
- `true`: Better robustness to slight pose shifts.
- `false`: More sensitive to shift/rotation.

### Indicator LED Enabled (`indicator_led.enabled`)
- Purpose: Enable GPIO pass/fail LED signaling on Pi.
- Does not change scoring logic.

## Practical Tuning Order

1. Start with `mask_only` mode.
2. Set capture exposure (`shutter_us`) for stable contrast.
3. Set `threshold_value` so mask shape matches real feature.
4. Tune `min_required_coverage` and `max_outside_allowed_ratio`.
5. Tune `min_section_coverage` for local defect sensitivity.
6. Enable optional SSIM/MSE/ML gates only after mask baseline is stable.

## Quick Symptoms and Fixes

- Too many false rejects:
  - Lower `min_required_coverage`.
  - Raise `max_outside_allowed_ratio`.
  - Lower `min_section_coverage`.

- Missed obvious defects:
  - Raise `min_required_coverage`.
  - Lower `max_outside_allowed_ratio`.
  - Raise `min_section_coverage`.

- Noisy masks:
  - Increase `threshold_value`.
  - Improve lighting/camera exposure consistency.

- Dark captures:
  - Increase `shutter_us`.
  - Verify lighting and camera gain profile.

# Registration-First Redesign Plan

Bring commissioning and inspection under control by making registration the first-class contract for each project. The immediate goal is not a big runtime rewrite. The goal is to establish the data model and rollout path so each later slice is testable, Pi-safe, and reversible.

## Stages

1. Foundation contracts
- Add explicit registration schema to project config and stored reference metadata.
- Preserve current moments-based runtime behavior while the richer model is introduced.
- Define anchors, datum frame, search margin, transform model, and registration quality gates as durable per-project settings.

2. Registration engine expansion
- Keep the current moments flow as the compatibility fallback.
- Add a dedicated registration engine that can execute staged strategies such as single-anchor translation, paired-anchor similarity, and refined rigid registration.
- Separate registration failure from feature/insert failure in runtime results.

3. Datum-relative measurement plan
- Express measurements in part coordinates after registration, not just image coordinates inside the ROI.
- Move edge, width, and center checks onto datum-relative measurement definitions so tolerance behavior follows the part, not the camera placement.
- Keep thresholds configurable per project.

4. Commissioning workflow redesign
- Add commissioning steps for anchor placement, search windows, datum confirmation, and expected transform validation.
- Treat golden reference capture as baseline registration capture, not just mask baking.
- Keep operator UI explicit about what is being defined: anchor, datum, feature region, or approval sample.

5. Runtime decision and diagnostics cleanup
- Report registration confidence, residual error, and selected strategy alongside pass/fail results.
- Distinguish these outcomes in diagnostics: registration drift, placement error, missing insert, feature defect, and configuration mismatch.
- Reuse the same registration summary in replay and dashboard views.

6. Validation and Pi rollout
- Add focused tests for schema normalization, metadata persistence, registration strategy selection, and datum-relative measurement behavior.
- Run replay validation on existing samples before enabling any new registration mode by default.
- Gate Pi rollout behind smoke checks for commissioning, training, replay, and production loop stability.

## Current Slice

The current implementation has completed the foundation contract and started the registration engine stage:

- config defaults now carry registration-first commissioning structure
- reference metadata now persists registration settings
- mismatch detection now reports registration drift in addition to ROI/threshold drift
- shared config/editor schema now exposes scalar registration settings
- inspection runtime now passes through a dedicated registration engine boundary that reports runtime mode, requested strategy, fallback reason, and registration summary
- active runtime now supports `anchor_translation`, `anchor_pair`, and `rigid_refined` in addition to the compatibility `moments` path
- registration results now expose structured transform summaries and observed-anchor coordinates so datum-relative measurements can build on explicit registration outputs
- section width, center, and edge gates now consume datum-frame measurements derived from the registration transform when registration succeeds, with aligned-mask fallback retained for unsupported runtime modes
- replay output now carries the selected reference path plus compact per-candidate registration and measurement-frame summaries so stored-image validation can inspect multi-reference selection behavior
- registration quality gates now participate in runtime acceptance, with explicit registration-failure reporting separated from part-level defect failure in replay and diagnostics

Active runtime behavior is still rollout-controlled: `moments`, `anchor_translation`, `anchor_pair`, and `rigid_refined` are available through `alignment.mode`; datum-relative section width/center/edge gating is now active; registration quality gates can now reject otherwise-aligned transforms before part-level inspection is trusted; replay validation has been exercised on the stored `5096v2.0` dataset with datum-frame metrics active across all 24 captures and 11 passes after the section-occupancy fix; commissioning status now surfaces registration readiness and golden-reference capture now stamps a baseline registration contract into reference metadata. The config editor now includes a dedicated registration setup dialog for anchor placement, search-window definition, datum confirmation, and expected-transform review, and commissioning treats missing or stale baseline capture as an actionable workflow stage rather than passive metadata drift.
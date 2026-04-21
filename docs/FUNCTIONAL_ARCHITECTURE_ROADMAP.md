# Functional Architecture Roadmap

This document combines the near-term pilot-functional work with the longer-term architecture needed to support a broad family of molding and manufacturing inspections without repeatedly redesigning the system.

It is intentionally centered on runtime function, not operator permissions or UI policy.

## Scope

Current first-use situations:

- light-pipe geometry
- white pad print on black part

Expected future scope:

- additional molded-feature geometry checks
- print checks with different defect classes
- presence/absence checks
- seat/placement checks
- contamination and foreign-material checks
- multiple focused inspections on the same part during one cycle

## Current Decision

Keep the recently added readiness/operator-sidecar work in place.

Why:

- it is largely isolated from the core runtime path
- it does not materially change the decision logic inside the inspection pipeline
- removing it now would cost time without moving functional readiness forward

The architectural work below is therefore aimed at:

- `inspection_system/app/preprocessing_utils.py`
- `inspection_system/app/registration_engine.py`
- `inspection_system/app/inspection_pipeline.py`
- `inspection_system/app/scoring_utils.py`
- `inspection_system/app/feature_measurement_utils.py`

## Core Problem To Solve Once

The current runtime is still shaped like one inspection path with more options being attached to it.

That is enough for the immediate pilot, but it is not the right shape for a future where one part may need several focused inspections with different segmentation, measurements, and gates.

The architecture therefore needs to shift from:

- one pipeline with optional metrics and ad hoc dict fields

to:

- one shared part program
- multiple inspection lanes per part
- per-lane segmentation, measurement, and gating
- one final aggregation layer

## Architectural Principles

1. Shared acquisition and registration happen once per part.
2. Each inspection lane evaluates one focused concern.
3. Segmentation, measurement, and gating are lane-specific.
4. The orchestration layer coordinates; it does not own every rule.
5. Results move between stages as typed contracts, not expanding dict blobs.
6. New inspection scope should usually mean adding a lane or strategy, not rewriting the pipeline.

## Target Runtime Model

### Shared stages

- acquisition
- common image prep
- registration
- shared frame bundle generation

### Lane stages

Each lane receives the shared registered part context and runs:

- segmentation strategy
- measurement extractors
- gate evaluators
- lane-specific result interpretation

### Aggregation stage

The system combines lane results into:

- part pass/fail/review
- failed lane list
- lane-level evidence for replay and debugging

## Typed Contracts To Introduce

The following contracts should be introduced before substantial new functional logic is added.

### Shared contracts

- `FrameBundle`
  Raw image, ROI image, grayscale image, registration image variants, and any reusable image intermediates.

- `RegistrationAssessment`
  Wraps or aliases the existing registration result with a stable contract for downstream consumers.

- `PartContext`
  Shared runtime context after registration, containing transforms, datum frame, reference assets, and shared imagery.

### Lane contracts

- `SegmentationResult`
  Lane-specific mask or pixel-selection output plus segmentation diagnostics.

- `MeasurementBundle`
  Lane-specific structured measurements.

- `GateCheck`
  One evaluated gate with threshold, observed value, pass/fail, and summary.

- `GateDecision`
  Aggregate lane decision, including all failed checks and advisory checks.

- `LaneResult`
  Full lane output combining segmentation summary, measurements, and gate decision.

### Final contract

- `InspectionOutcome`
  Part-level final result with lane list, aggregated status, and debug metadata.

## Inspection Program Model

The system should add an inspection-program configuration layer above loose config keys.

An inspection program defines:

- shared registration settings
- enabled lanes
- lane execution order
- lane aggregation policy
- pilot defect scope per lane

Each lane definition should contain:

- `lane_id`
- `lane_type`
- ROI or datum window selector
- segmentation strategy name
- measurement family names
- gate profile or explicit thresholds
- defect labels in scope
- whether the lane is authoritative or advisory

### Examples

#### Near-term program for a molded part with print

- lane A: light-pipe geometry
- lane B: white pad print completeness and shift

#### Future program on the same part

- lane A: molded feature position
- lane B: print registration
- lane C: clip presence
- lane D: contamination check near sealing edge

## Module Breakdown

### Keep and narrow existing responsibilities

- `preprocessing_utils.py`
  Keep ROI extraction and common image preparation only.

- `registration_engine.py`
  Keep runtime registration as the shared registration subsystem.

- `feature_measurement_utils.py`
  Keep feature-family measurement extraction and extend it with focused measurement families.

- `inspection_pipeline.py`
  Reduce to orchestration and result assembly.

### Add new module families

- `inspection_models.py`
  Dataclasses or typed result contracts.

- `inspection_program.py`
  Program and lane definitions plus config loading.

- `segmentation/`
  Lane segmentation strategies.

- `gates/coverage_gates.py`
  Coverage and allowed-area checks.

- `gates/geometry_gates.py`
  Section geometry checks.

- `gates/feature_gates.py`
  Feature-family and position-specific checks.

- `gates/anomaly_gates.py`
  SSIM, MSE, anomaly-score, and future image-similarity checks.

- `lanes/`
  Lane executors that bind segmentation, measurement, and gating together.

- `aggregation.py`
  Part-level combination of lane results.

## Near-Term Functional Priorities

### Priority 1 - Make geometry lanes real gates

The current pipeline computes `feature_measurements` and `feature_position_summary`, but final gating still flows mainly through legacy mask and section metrics.

For pilot readiness, the first architectural win is to make feature-family measurements authoritative for the geometry lane.

This includes thresholds for:

- `dx_px`
- `dy_px`
- `radial_offset_px`
- `pair_spacing_delta_px`

Legacy section-center logic should remain as fallback or compatibility behavior, not the long-term main geometry authority.

### Priority 2 - Split gate evaluation by domain

Do not keep extending one universal `evaluate_metrics(...)` block.

Instead:

- coverage gates evaluate print/mask coverage
- geometry gates evaluate section-based shape drift
- feature gates evaluate datum-relative feature families
- anomaly gates evaluate similarity/anomaly signals

Then combine those gate results in one thin layer.

### Priority 3 - Add segmentation strategies as first-class runtime choices

The current threshold-first front end should become a strategy family so that future recipes do not force branching inside one `make_binary_mask(...)` function.

Initial strategy names:

- `binary_threshold`
- `binary_threshold_inverted`

Planned later strategies:

- `local_contrast_threshold`
- `print_roi_threshold`
- `color_range_mask`

### Priority 4 - Add inspection programs and lanes before adding more defect types

Before the third major inspection type is added, the runtime should support multiple lanes explicitly.

That ensures the architecture is prepared for:

- multiple focused checks on one part
- different segmentation methods per lane
- advisory versus authoritative lanes

## Phased Implementation Plan

### Phase A - Stabilize contracts without changing behavior

Goal:

- introduce typed models and thin wrappers around current dict outputs

Work:

- add `inspection_models.py`
- define `RegistrationAssessment`, `MeasurementBundle`, `GateDecision`, `InspectionOutcome`
- wrap current pipeline outputs into those models while preserving current runtime behavior

Why first:

- it reduces future refactor pain immediately
- it prevents more optional keys from spreading through the pipeline

### Phase B - Decompose gating while preserving external behavior

Goal:

- move logic out of one gate block into domain-specific evaluators

Work:

- add `gates/coverage_gates.py`
- add `gates/geometry_gates.py`
- add `gates/anomaly_gates.py`
- keep `scoring_utils.py` as a compatibility entrypoint that delegates

Why second:

- it creates modular seams before new feature-family gate logic is added

### Phase C - Make feature-family geometry authoritative

Goal:

- promote molded-part feature families from diagnostics to real gating

Work:

- add `gates/feature_gates.py`
- define explicit threshold schema for family metrics
- wire feature gates into the combined decision
- retain legacy section-center behavior as compatibility fallback

Why third:

- this is the highest-value functional change for the light-pipe pilot

### Phase D - Introduce segmentation strategy family

Goal:

- remove recipe-specific front-end branching pressure from `preprocessing_utils.py`

Work:

- keep common ROI and shared image prep in `preprocessing_utils.py`
- add segmentation strategy modules under `segmentation/`
- wire current threshold modes through the new strategy family

Why fourth:

- it prepares the architecture for print-focused and future specialized lanes

### Phase E - Introduce inspection programs and lane execution

Goal:

- support multiple focused inspections on the same part cleanly

Work:

- add `inspection_program.py`
- define lane config schema
- add lane runner interface
- update `inspection_pipeline.py` to orchestrate shared stages and per-lane execution

Why fifth:

- this is the point where the architecture becomes broadly extensible rather than merely modularized

### Phase F - Evidence-driven validation and pilot tuning

Goal:

- prove runtime behavior on saved challenge data and real variation

Work:

- replay challenge sets through lane-aware results
- tune false reject and false accept behavior by lane
- validate registration robustness with real part variation

Why last:

- the validation should exercise the intended architecture rather than a temporary one

## Exact First Refactor Slice

The first slice should be deliberately small.

### Slice 1

- add `inspection_models.py`
- wrap the current pipeline output into typed models without changing external runtime behavior
- split `evaluate_metrics(...)` internals into delegated coverage, geometry, and anomaly gate helpers
- keep the public API stable

This slice should not yet add lane execution.

It should only create the shape that prevents future logic from being stuffed into the same two files.

## Architectural Non-Goals For The First Slice

Do not do these in the first refactor slice:

- no giant rewrite of registration
- no plugin framework
- no abstract factory layers with no immediate use
- no attempt to redesign every future inspection type up front
- no forced UI or operator-flow redesign

## Definition Of Success

This architecture work is successful if:

1. The current runtime still works for the pilot recipes.
2. New defect logic stops accumulating as optional dict fields.
3. The geometry lane can become authoritative without destabilizing print logic.
4. A future additional lane can be added without rewriting the pipeline.
5. One part can eventually support several focused inspections in one run.

## Summary

The goal is not just to modularize the current pilot.

The goal is to move from a single expandable inspection path to a shared-part, multi-lane inspection framework that can support:

- light-pipe geometry now
- white pad print on black part now
- broader molding and manufacturing checks later
- several focused inspections on the same part later

That is the architecture work worth doing once.
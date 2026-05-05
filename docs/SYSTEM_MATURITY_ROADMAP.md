# System Maturity Roadmap

This document is the high-level execution roadmap for maturing Beacon from a first-use, print-centered inspection core into one reusable inspection platform that can host mold-specific or part-specific inspection programs.

It complements, rather than replaces, the lower-level design detail in `docs/FUNCTIONAL_ARCHITECTURE_ROADMAP.md`.

## Purpose

The point of this roadmap is to keep the project moving toward the stated system intent while preserving the operational value that already exists.

This is not a future-state wish list. It is the current direction of the project and the sequence for getting there without destabilizing the pilot-ready surfaces that already work.

## System Intent

Beacon is intended to be:

- one reusable controlled-imaging inspection system
- one shared runtime, operator flow, and commissioning model
- many mold-specific or part-specific inspection programs
- a visual quality inspection platform, not a metrology system
- a low-training, recipe-driven visual exception detection system for molded parts and related controlled-imaging checks

Beacon is not intended to be:

- a separate bespoke application for every mold
- a runtime whose core logic is permanently defined by one first-use print inspection
- a system that can only broaden by bolting more options onto one original scoring path
- a giant defect-classification AI that must know every defect class before it can be useful

## Inspection philosophy

The system should be built on this operating assumption:

- good parts usually look similar under controlled conditions
- bad parts may be visually diverse

So the primary question is:

- does this part look acceptably like the known-good family under the expected imaging conditions?

not:

- can the software perfectly name every possible defect class?

This makes Beacon a visual guard system first:

- controlled acquisition
- recipe-controlled lighting and timing
- image-quality validation
- registration into a shared frame
- lane-based defect-family checks
- anomaly or similarity evidence where helpful
- conservative pass/fail/review/invalid decisions
- operator review and replay-backed refinement

## Current Reality

The project already has strong operational foundations:

- project and recipe management
- operator dashboard workflows
- golden-reference capture and registration commissioning
- replay validation and supervised pilot readiness
- training and review flows
- Raspberry Pi deployment and optional RS-485 / Modbus IO

The main maturity gap is in the inspection core.

The core is still too strongly shaped by a single-path reference-mask workflow. That makes it harder than it should be to support peer inspection concerns such as molded-feature presence, seat / placement checks, short shots, contamination, and surface-defect families.

## Guiding Rules

1. Preserve the working operator and pilot shell while the core changes.
2. Replace first-use assumptions in the core; do not just hide them behind more settings.
3. Add peer inspection capabilities, not more branches inside one legacy path.
4. Make new defect families explicit in runtime contracts, config, and evidence.
5. Keep rollout reversible: each phase should leave the repo in a usable state.
6. Keep `INVALID_CAPTURE` separate from `FAIL`; a bad image is not a bad part.
7. Prefer low-training good-family modeling plus anomaly evidence over defect-classification-by-default.
8. Treat acquisition, lighting, and image-quality validation as core architecture, not station-side details.

## High-Level Roadmap

### Stage 0 - Preserve the current stable baseline

Objective:

- keep a stable, pilot-capable baseline while redesign work proceeds

Outcomes:

- stable tagged baseline for current deployment state
- current workflows remain runnable while the core evolves
- architectural work can proceed without fear of losing the known-good path

Exit criteria:

- the team can point to a known stable baseline
- redesign work is understood to happen above that baseline, not by erasing it

### Stage 1 - Define the core contracts

Objective:

- stop the inspection pipeline from expanding through ad hoc dict fields and one-off conditionals

Outcomes:

- explicit shared contracts for registration, measurements, gates, lane results, and inspection outcome
- a stable shape for new logic to plug into
- reduced pressure to keep extending the legacy path in place

Exit criteria:

- the current pipeline result shape is wrapped in stable typed contracts
- new logic no longer requires scattering optional fields across unrelated modules

### Stage 2 - Break the single scoring path into domains

Objective:

- separate defect reasoning by domain instead of forcing all decisions through one universal evaluation block

Outcomes:

- independent gate families for coverage, geometry, feature-family logic, and anomaly / similarity logic
- thinner orchestration layer in the pipeline
- easier explanation of why a part passed, failed, or needs review
- a clearer boundary between part-quality failure and image / setup failure

Exit criteria:

- domain-specific gate evaluators exist and drive the combined decision
- current recipes still run without regressions in their expected behavior

### Stage 3 - Make at least one non-print inspection concern first-class

Objective:

- prove the system can support a genuine peer inspection concern rather than extending the print-centered path again

Candidate first-class concerns:

- molded-feature position
- molded-feature presence / absence
- short-shot detection in defined regions
- seat / placement verification

Outcomes:

- one non-print concern is implemented as a real first-class inspection capability
- evidence, gating, and failure causes are specific to that concern
- the core stops treating the original print workflow as the universal model

Exit criteria:

- one production-relevant non-print inspection can run end to end through config, replay, and operator flow
- failure reporting is concern-specific rather than a generic mask score explanation

### Stage 4 - Introduce real inspection programs and peer lanes

Objective:

- move from one configurable inspection path to one shared part program with multiple focused inspection lanes

Outcomes:

- explicit inspection-program model
- per-lane segmentation, measurement, and gate configuration
- shared acquisition and registration with multiple lane executions per part
- authoritative versus advisory lanes supported intentionally

Exit criteria:

- one part can run multiple focused checks in one cycle without pipeline rewrites
- adding a lane does not require changing unrelated lane logic

### Stage 5 - Align config and operator surfaces with the new core

Objective:

- make the UI and config model reflect inspection programs and lanes instead of a single global tuning surface

Outcomes:

- project config evolves from one global inspection block toward shared settings plus lane-specific settings
- acquisition and lighting configuration evolve toward recipe-defined behavior where needed
- commissioning and config editing can express the new inspection structure
- operator evidence is organized by lane or defect concern where appropriate

Exit criteria:

- operators and engineers can see which inspection concern failed and why
- config surfaces no longer assume one global segmentation and threshold stack is the whole system

### Stage 6 - Validate robustness against real defect families

Objective:

- prove the redesigned core improves robustness on realistic molded-part defect behavior

Defect families to validate deliberately:

- missing features
- short shots
- seat / placement issues
- contamination or extra material
- print / marking issues where applicable
- surface-defect families such as splay where the evidence model supports them

Outcomes:

- replay and pilot validation organized by inspection concern, not just total pass/fail
- clearer false-reject and false-accept analysis
- better explanation of weak areas still needing maturity work
- stronger evidence for when a part should be `REVIEW` versus hard `FAIL`

Exit criteria:

- the team can name which defect families are strong, emerging, or not yet ready
- robustness claims are based on evidence by concern, not on one aggregate score

## What Changes First And What Stays Stable

Should change first:

- core contracts
- gate decomposition
- first-class non-print inspection logic
- lane-aware execution model

Should stay stable unless a redesign step requires otherwise:

- project management model
- operator dashboard entrypoints
- replay workflow
- commissioning intent
- Raspberry Pi deployment and IO integration surfaces

## Anti-Patterns To Avoid

Do not treat these as progress:

- adding more global settings to force new defect types into the old scoring path
- describing the broader platform as intent while continuing to code only for the first-use recipe
- adding UI polish that hides core architectural limitations
- introducing abstract plugin systems before the runtime has even proven peer lane types
- destabilizing the pilot-ready shell to chase purity in the core
- assuming the system should classify every defect before it can be useful in production

## Decision Gates

Use these questions during planning and review:

1. Does this change reduce dependence on the original print-centered core, or deepen it?
2. Does this add a peer inspection capability, or just another legacy-path option?
3. Does this make failure causes more specific and explainable?
4. Does this preserve the working operational shell while the core matures?
5. Would this still make sense if the next mold had no print inspection at all?

## Success Definition

This roadmap is succeeding when the project can honestly say:

- Beacon is one inspection platform, not one historic recipe with extra settings.
- A new mold or part program does not require a bespoke code path by default.
- Non-print inspection concerns are first-class peers in the runtime.
- Operators and engineers can understand inspection outcomes by concern, not just by aggregate score.
- The system is becoming more robust because the core is changing shape, not because more tuning was layered onto the old path.
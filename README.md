# Beacon AI Inspection Camera

Raspberry Pi–based visual inspection system for project-based commissioning, supervised pilot rollout, replay validation, and production-side inspection under controlled imaging.

## System intent

This system is intended to be **one reusable inspection platform** that can host mold-specific or part-specific inspection programs without turning each new mold into a separate bespoke application.

That intent is part of the current state of the project, not a deferred future-state claim. The codebase is in an ongoing maturity effort to move from a first-use, print-centered inspection core toward a more robust configurable visual quality inspection system for molded parts and related controlled-imaging checks.

The goal is:

- one shared runtime, operator flow, and commissioning model
- project or recipe-specific inspection programs per mold / part / defect scope
- multiple inspection concerns handled as peer capabilities rather than extensions of one original print-inspection path
- strong visual quality inspection under controlled imaging, without positioning the system as a metrology platform

The guiding inspection philosophy is:

- **low-training, recipe-driven visual exception detection for molded parts**
- model acceptable visual appearance and known-good variation first
- use registration, lane-specific gates, and anomaly scoring to flag meaningful deviations
- prefer conservative `PASS` / `FAIL` / `REVIEW` / `INVALID_CAPTURE` decisions over pretending the system can confidently name every possible defect
- treat bad image quality or bad setup as different from a bad part

## Current truth

This repository currently contains a working inspection system centered on:

- project-based recipe management and operator dashboard workflows
- golden-reference capture, registration commissioning, and config tuning
- live inspection, replay validation, and supervised pilot readiness checks
- interactive training and review flows for borderline decisions
- optional RS-485 / Modbus trigger and indicator integration on Raspberry Pi

The current implementation is strongest as a **controlled-imaging inspection platform for focused molded-feature and print / marking checks**. The system intent is broader than that today, but the inspection core is still maturing away from its original single-path, print-centered design.

That means the active direction is not "train one AI model to know every defect". The active direction is to combine controlled acquisition, good-family references, registration, defect-family lanes, anomaly evidence, and operator review into a robust visual guard system.

## Current repository shape

- `inspection_system/app/capture_test.py` — CLI entrypoint for dashboard, capture, reference, project, and readiness flows
- `inspection_system/app/operator_dashboard.py` — unified operator launch surface
- `inspection_system/app/config_editor_page.py` — commissioning and config editor UI, including registration and IO setup
- `inspection_system/app/interactive_training.py` — human-in-the-loop training and review workflow
- `inspection_system/app/replay_inspection.py` — replay runner for saved-image validation
- `inspection_system/config/camera_config.json` — active capture, inspection, and IO tuning
- `inspection_system/projects/` — per-project configs, references, and logs
- `samples/` — starter folder structure for replay and collected test datasets

## Raspberry Pi connection

If the Pi was restarted or you are reconnecting after downtime, use the connection runbook:

- `docs/PI_CONNECTION.md`

Quick reconnect from Windows PowerShell:

```powershell
ssh pi@pi-inspect
```

## Recommended workflow

### 1. Set up Projects (New!)

For managing multiple products or inspection setups:

```bash
# Launch unified operator dashboard
python3 inspection_system/app/capture_test.py dashboard

# Create a new project
python3 inspection_system/app/capture_test.py create-project "Widget_A" "Widget A inspection setup"

# Switch between projects
python3 inspection_system/app/capture_test.py switch-project "Widget_A"

# List all projects
python3 inspection_system/app/capture_test.py list-projects

# Launch GUI project manager (requires tkinter)
python3 inspection_system/app/capture_test.py project-manager
```

The operator dashboard provides a single window for:
- capture-only runs
- reference creation
- live inspect runs
- launching interactive training
- quick project switching and project creation
- recent training-log summaries for the active project

Each project maintains its own:
- Configuration settings (`camera_config.json`)
- Reference images (`golden_reference_mask.png`, `golden_reference_image.png`)
- Training logs and session data

### 2. Capture-only test

```bash
python3 inspection_system/app/capture_test.py capture
```

### 2a. Check Supervised Pilot Readiness

Before a supervised floor pilot, run the readiness report for the active project:

```bash
python3 inspection_system/app/capture_test.py pilot-readiness
```

This reports:

- active project and reference state
- commissioning readiness
- saved challenge-set counts for `tuning`, `validation`, and `regression`
- open blocking issues before first supervised pilot launch

### 3. Set the golden reference

```bash
python3 inspection_system/app/capture_test.py set-reference
```

### 4. Interactive Training Mode

```bash
pip install pygame  # Required for GUI
python3 inspection_system/app/capture_test.py train
```

This mode provides a graphical interface for human-in-the-loop training:

- **Real-time Display**: Shows captured images with color-coded borders
  - 🟢 **Green border**: Automatic pass
  - 🔴 **Red border**: Automatic fail  
  - 🟡 **Yellow border**: Flagged for human review (close to thresholds)

- **Detailed Descriptions**: Human-readable explanations of inspection results:
  - **Pass**: "✓ APPROVED: Sample meets all quality requirements"
  - **Coverage Issues**: "✗ REQUIRED COVERAGE: Missing 15.2% of required features"
  - **Contamination**: "✗ OUTSIDE ALLOWED: 3.1% excess material in restricted areas"
  - **Anomalies**: "✗ ANOMALY DETECTED: High anomaly score (0.723) suggests defects"

- **Interactive Feedback**: Click buttons to provide feedback:
  - **APPROVE**: Accept borderline samples to expand acceptable range
  - **REJECT**: Reject samples to tighten quality standards
  - **REVIEW**: Flag samples for later manual review

- **Training Logs**: Comprehensive session logging in `inspection_system/logs/`:
  - Timestamped decisions and feedback
  - Full metric details for each sample
  - Session summaries and threshold suggestions
  - Training data for analysis and improvement

- **Adaptive Learning**: System analyzes feedback to suggest threshold adjustments
- **Session Tracking**: Real-time progress updates and final summaries

### 5. View Training Logs

```bash
python3 inspection_system/app/log_viewer.py --recent 20
```

Review your training sessions and analyze decision patterns:

- **Summary Statistics**: Overall approval rates and session counts
- **Recent Decisions**: Last N training decisions with full details
- **Session Analysis**: Performance tracking across multiple training sessions

### 6. Replay a saved image

```bash
python3 inspection_system/app/replay_inspection.py inspect-file samples/good/example.jpg
```

### 5. Replay a whole folder

```bash
python3 inspection_system/app/replay_inspection.py inspect-folder samples/good
```

### 6. Run a sandboxed training episode from collected test images

Use the dashboard's `Collect Test Images` flow to create session folders with `captures.jsonl`, then run a local diagnostic episode without mutating the live project:

```bash
python3 -m inspection_system.app.dataset_diagnostics ./inspection_system/projects/MyProject/test_data --duplicates 3 --update-every 5
```

The runner will:

- load collected session manifests under the provided path
- replay `tuning` samples as simulated training feedback in a temporary sandbox copy of the project config and references
- apply threshold updates and commit cycles during the episode
- evaluate `validation` and `regression` samples afterward
- write a JSON report with false-reject, false-accept, and invalid-capture miss rates under a `diagnostics/` folder

## Supervised Pilot Workflow

The staged floor-pilot runbook is documented in:

- `docs/SUPERVISED_PILOT_RUNBOOK.md`
- `docs/FUNCTIONAL_ARCHITECTURE_ROADMAP.md`

Recommended execution order:

1. Switch to the correct project for the recipe.
2. Run `pilot-readiness`.
3. Run the learning pass with controlled challenge parts.
4. Re-run `pilot-readiness`.
5. Launch the first floor pilot only with engineering present.

## Project Management

The system supports multiple inspection projects, allowing you to maintain separate configurations and reference images for different products or inspection setups.

### Directory Structure

```
~/inspection_system/
├── config/
│   ├── camera_config.json          # Global fallback config
│   └── projects.json               # Project registry
├── projects/                       # Project-specific data
│   ├── project_a/
│   │   ├── config/
│   │   │   └── camera_config.json  # Project A config
│   │   ├── reference/
│   │   │   ├── golden_reference_mask.png
│   │   │   └── golden_reference_image.png
│   │   └── logs/                   # Project A training logs
│   └── project_b/
│       ├── config/
│       ├── reference/
│       └── logs/
└── reference/                      # Legacy global references
```

### GUI Project Manager

Launch the graphical project manager:

```bash
python3 inspection_system/app/capture_test.py project-manager
```

Features:
- **Create new projects** with descriptions
- **Switch between projects** instantly
- **Export/import projects** as ZIP files for backup/sharing
- **Delete projects** with confirmation
- **View project details** (creation date, description)

### Command-Line Project Management

```bash
# Create project
python3 inspection_system/app/capture_test.py create-project "MyProject" "Description"

# Switch to project
python3 inspection_system/app/capture_test.py switch-project "MyProject"

# List projects
python3 inspection_system/app/capture_test.py list-projects

# All other commands (capture, inspect, train) work within the current project
```

### Project Isolation

Each project maintains completely separate:
- **Configuration**: Camera settings, thresholds, ROI definitions
- **Reference Images**: Golden reference mask and image
- **Training Data**: Logs, session data, and learned parameters
- **Debug Images**: Saved inspection results and analysis images

This allows you to:
- Quickly switch between different product lines
- Maintain different inspection standards for various components
- Share project configurations between systems
- Archive completed projects for future reference

## Result states

Replay mode distinguishes between:

- `PASS`
- `FAIL`
- `INVALID_CAPTURE`
- `CONFIG_ERROR`

`INVALID_CAPTURE` is intended for unreadable images, bad ROI configuration, missing reference assets, or other conditions where the image should not be treated as a failed part.

## Sample dataset structure

```text
samples/
  good/
  bad/
  borderline/
  invalid_capture/
```

Use these folders to build a replayable regression set so software progress does not depend on having physical parts at the bench.

## Dependencies

Minimum runtime dependencies on the Pi:

- Python 3
- `rpicam-still`
- OpenCV (`python3-opencv`)
- NumPy (`python3-numpy`)
- optional: `RPi.GPIO`

Install OpenCV + NumPy on Raspberry Pi OS with:

```bash
sudo apt install -y python3-opencv python3-numpy
```

## Current limitations

- Current implementation is still concentrated in one main script.
- Runtime-generated debug files were committed in the first capture and should be cleaned out in a follow-up pass.
- Replay mode is added without a full module refactor yet.
- The project is still early in architecture maturity.

## Intended direction

Mature direction is a recipe-driven machine vision platform with:

- standardized inspection head
- fixture-specific lighting / imaging geometry
- recipe-defined task behavior
- centralized decision discipline
- traceable, replayable outcomes
- redeployable short-run use first, dedicated per-press deployment later where ROI proves out

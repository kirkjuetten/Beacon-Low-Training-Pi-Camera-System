# Beacon AI Inspection Camera

Raspberry Pi–based visual inspection prototype for localized print inspection under controlled imaging.

## Current truth

This repository currently contains a working narrow prototype centered on:

- ROI-based binary masking
- golden reference comparison
- limited angle / shift alignment
- threshold-based scoring
- optional GPIO pass/fail indicators

The current implementation is strongest as **localized print / marking inspection**, not yet as a generalized multi-recipe inspection platform.

## Current repository shape

- `inspection_system/app/capture_test.py` — live capture, reference creation, and inspection kernel
- `inspection_system/app/replay_inspection.py` — replay runner for saved-image testing
- `inspection_system/config/camera_config.json` — active capture / inspection tuning
- `inspection_system/reference/` — reference mask assets
- `samples/` — starter folder structure for replay datasets

## Recommended workflow

### 1. Capture-only test

```bash
python3 inspection_system/app/capture_test.py capture
```

### 2. Set the golden reference

```bash
python3 inspection_system/app/capture_test.py set-reference
```

### 3. Live inspection

```bash
python3 inspection_system/app/capture_test.py inspect
```

### 4. Replay a saved image

```bash
python3 inspection_system/app/replay_inspection.py inspect-file samples/good/example.jpg
```

### 5. Replay a whole folder

```bash
python3 inspection_system/app/replay_inspection.py inspect-folder samples/good
```

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

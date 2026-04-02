# Sample replay dataset

Use this folder to build a replayable image set for tuning and regression checks.

## Suggested structure

- `good/` — known good captures
- `bad/` — known bad or synthetic-bad captures
- `borderline/` — questionable cases worth review
- `invalid_capture/` — glare, blur, bad placement, clipped frame, exposure problems

## Goal

Software progress should not depend on having physical parts available every time.
Build a saved-image set here and replay it after configuration or code changes.

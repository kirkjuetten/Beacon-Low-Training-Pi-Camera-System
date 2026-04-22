#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Beacon-Low-Training-Pi-Camera-System}"
LOG_FILE="${LOG_FILE:-$HOME/beacon-dashboard.log}"

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"

cd "$REPO_DIR"

# Run the operator dashboard and keep a simple launch log for troubleshooting.
python3 -m inspection_system.app.capture_test dashboard >>"$LOG_FILE" 2>&1

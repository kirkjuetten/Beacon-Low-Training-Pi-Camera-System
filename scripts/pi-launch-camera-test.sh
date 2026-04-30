#!/usr/bin/env bash
# Beacon Camera Test - friendly desktop launcher.
#
# Runs three diagnostics and prints PASS/FAIL for each:
#   1. Sensor enumeration via rpicam-hello --list-cameras.
#   2. Single still capture via rpicam-still (writes /tmp/beacon_camera_probe.jpg).
#   3. Project-level capture round-trip via capture_test capture.
#
# The output is also appended to ~/beacon-camera-test.log for later review.
set -uo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Beacon-Low-Training-Pi-Camera-System}"
LOG_FILE="${LOG_FILE:-$HOME/beacon-camera-test.log}"
PROBE_IMAGE="/tmp/beacon_camera_probe.jpg"

cd "$REPO_DIR" 2>/dev/null || true

{
    echo "==============================================="
    echo "Beacon Camera Test - $(date)"
    echo "==============================================="
} | tee -a "$LOG_FILE"

pass=0
fail=0
note() { echo "[$1] $2" | tee -a "$LOG_FILE"; }

# --- Step 1: enumerate sensors -------------------------------------------------
echo
echo "Step 1: detect camera sensor"
list_output=$(rpicam-hello --list-cameras 2>&1 || true)
echo "$list_output" >>"$LOG_FILE"
if echo "$list_output" | grep -qE '^[[:space:]]*[0-9]+ : '; then
    note PASS "Camera sensor detected on CSI bus."
    pass=$((pass + 1))
else
    note FAIL "No camera detected. Check the CSI ribbon (both ends), ribbon type, and orientation."
    fail=$((fail + 1))
fi

# --- Step 2: direct still capture ---------------------------------------------
echo
echo "Step 2: capture a still image"
rm -f "$PROBE_IMAGE"
if timeout 15 rpicam-still --nopreview -o "$PROBE_IMAGE" --timeout 1500 >>"$LOG_FILE" 2>&1 \
   && [ -s "$PROBE_IMAGE" ]; then
    bytes=$(stat -c%s "$PROBE_IMAGE")
    note PASS "Captured $bytes bytes to $PROBE_IMAGE."
    pass=$((pass + 1))
else
    note FAIL "rpicam-still timed out or produced no image. See $LOG_FILE for libcamera output."
    fail=$((fail + 1))
fi

# --- Step 3: project capture ---------------------------------------------------
echo
echo "Step 3: project capture round-trip"
if [ -d "$REPO_DIR" ]; then
    if (cd "$REPO_DIR" && timeout 60 python3 -m inspection_system.app.capture_test capture) \
        >>"$LOG_FILE" 2>&1; then
        note PASS "capture_test capture succeeded."
        pass=$((pass + 1))
    else
        note FAIL "capture_test capture failed. See $LOG_FILE."
        fail=$((fail + 1))
    fi
else
    note FAIL "Repo not found at $REPO_DIR; skipping project capture."
    fail=$((fail + 1))
fi

echo
echo "-----------------------------------------------"
if [ "$fail" -eq 0 ]; then
    echo "Result: ALL $pass CHECKS PASSED"
    if command -v xdg-open >/dev/null 2>&1 && [ -f "$PROBE_IMAGE" ]; then
        xdg-open "$PROBE_IMAGE" >/dev/null 2>&1 &
    fi
else
    echo "Result: $fail FAILED, $pass PASSED"
    echo
    echo "If the camera was NOT detected (Step 1 failed):"
    echo "  - Confirm the right sensor mode is selected:"
    echo "      bash $REPO_DIR/scripts/pi-camera-select.sh status"
    echo "  - imx296 / most CSI cameras: 'sudo bash ... imx296' then reboot"
    echo "  - Sony IMX500 AI Camera:     'sudo bash ... imx500' then reboot"
    echo
    echo "If the camera was detected but capture timed out (Steps 2/3):"
    echo "  - Power off the Pi, lift the CSI latches at BOTH ends, fully"
    echo "    insert the ribbon square (silver side toward the HDMI ports"
    echo "    on the Pi), close the latches, and run this test again."
fi
echo "Log: $LOG_FILE"
echo
echo "Press ENTER to close this window."
read -r _ || true

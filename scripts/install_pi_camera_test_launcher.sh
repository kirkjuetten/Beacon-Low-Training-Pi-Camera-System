#!/usr/bin/env bash
# Install a "Beacon Camera Test" desktop icon next to the dashboard icon.
#
# The launcher opens a terminal window, runs scripts/pi-launch-camera-test.sh,
# and waits for the operator to read the results before closing.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Beacon-Low-Training-Pi-Camera-System}"
LAUNCH_SCRIPT="$REPO_DIR/scripts/pi-launch-camera-test.sh"
APP_DIR="$HOME/.local/share/applications"
DESKTOP_DIR="$HOME/Desktop"
APP_FILE="$APP_DIR/beacon-camera-test.desktop"
DESKTOP_FILE="$DESKTOP_DIR/Beacon Camera Test.desktop"

mkdir -p "$APP_DIR" "$DESKTOP_DIR"
chmod +x "$LAUNCH_SCRIPT"

# Pick a terminal that is actually installed; lxterminal ships with Pi OS,
# x-terminal-emulator is the Debian alternatives wrapper, xterm is the
# minimal fallback.
TERMINAL_CMD=""
for candidate in lxterminal x-terminal-emulator xterm; do
    if command -v "$candidate" >/dev/null 2>&1; then
        TERMINAL_CMD="$candidate"
        break
    fi
done

if [ -z "$TERMINAL_CMD" ]; then
    echo "No terminal emulator found (tried lxterminal, x-terminal-emulator, xterm)." >&2
    echo "Install one with: sudo apt install lxterminal" >&2
    exit 1
fi

case "$TERMINAL_CMD" in
    lxterminal)
        EXEC_LINE="lxterminal --title='Beacon Camera Test' -e /bin/bash $LAUNCH_SCRIPT"
        ;;
    x-terminal-emulator)
        EXEC_LINE="x-terminal-emulator -T 'Beacon Camera Test' -e /bin/bash $LAUNCH_SCRIPT"
        ;;
    xterm)
        EXEC_LINE="xterm -T 'Beacon Camera Test' -e /bin/bash $LAUNCH_SCRIPT"
        ;;
esac

cat >"$APP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Beacon Camera Test
Comment=Run a quick camera diagnostic
Exec=$EXEC_LINE
Path=$REPO_DIR
Icon=camera-photo
Terminal=false
Categories=Utility;
StartupNotify=true
EOF

cp "$APP_FILE" "$DESKTOP_FILE"
chmod +x "$APP_FILE" "$DESKTOP_FILE"

echo "Installed launcher: $APP_FILE"
echo "Desktop shortcut:  $DESKTOP_FILE"
echo "Double-click 'Beacon Camera Test' on the desktop to run the diagnostic."

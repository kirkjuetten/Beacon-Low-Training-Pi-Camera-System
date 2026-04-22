#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Beacon-Low-Training-Pi-Camera-System}"
LAUNCH_SCRIPT="$REPO_DIR/scripts/pi-launch-dashboard.sh"
APP_DIR="$HOME/.local/share/applications"
DESKTOP_DIR="$HOME/Desktop"
APP_FILE="$APP_DIR/beacon-inspection-dashboard.desktop"
DESKTOP_FILE="$DESKTOP_DIR/Beacon Inspection Dashboard.desktop"

mkdir -p "$APP_DIR" "$DESKTOP_DIR"
chmod +x "$LAUNCH_SCRIPT"

cat >"$APP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Beacon Inspection Dashboard
Comment=Launch Beacon operator dashboard
Exec=/bin/bash $LAUNCH_SCRIPT
Path=$REPO_DIR
Icon=applications-engineering
Terminal=false
Categories=Utility;
StartupNotify=true
EOF

cp "$APP_FILE" "$DESKTOP_FILE"
chmod +x "$APP_FILE" "$DESKTOP_FILE"

echo "Installed launcher: $APP_FILE"
echo "Desktop shortcut:  $DESKTOP_FILE"
echo "You can now launch from the desktop icon or app menu."

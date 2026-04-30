#!/usr/bin/env bash
# Beacon camera selector - one-command swap between supported sensors.
#
# Usage:
#   sudo bash pi-camera-select.sh imx296   # Global Shutter Camera (auto-detect)
#   sudo bash pi-camera-select.sh imx500   # Sony AI Camera (explicit overlay)
#   bash      pi-camera-select.sh status   # Show current selection (no sudo)
#
# Notes:
#   - Edits /boot/firmware/config.txt in-place inside a marked block. The first
#     run takes a one-time backup at config.txt.bak.camera-select.
#   - A reboot is required for the change to take effect.
#   - The imx296 (and most CSI cameras) are handled by camera_auto_detect=1.
#     The IMX500 AI Camera on Pi 4 is NOT handled by auto-detect and needs
#     dtoverlay=imx500 explicitly.
set -uo pipefail

CONFIG_FILE="${CONFIG_FILE:-/boot/firmware/config.txt}"
BACKUP_FILE="${CONFIG_FILE}.bak.camera-select"
BLOCK_BEGIN="# >>> beacon-camera-select >>>"
BLOCK_END="# <<< beacon-camera-select <<<"

usage() {
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

[[ $# -eq 1 ]] || usage
mode="$1"

case "$mode" in
    imx296|imx500|status) ;;
    -h|--help|help) usage ;;
    *) echo "Unknown mode: $mode" >&2; usage ;;
esac

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# --- status: read-only ---------------------------------------------------
if [[ "$mode" == "status" ]]; then
    echo "Config file: $CONFIG_FILE"
    if grep -q "$BLOCK_BEGIN" "$CONFIG_FILE"; then
        echo "Managed block:"
        sed -n "/$BLOCK_BEGIN/,/$BLOCK_END/p" "$CONFIG_FILE"
    else
        echo "No managed block present. Active camera-related lines:"
        grep -nE '^[[:space:]]*(#?[[:space:]]*)(camera_auto_detect|dtoverlay=imx)' \
            "$CONFIG_FILE" || echo "  (none)"
    fi
    echo
    if command -v rpicam-hello >/dev/null 2>&1; then
        echo "Currently detected:"
        rpicam-hello --list-cameras 2>&1 | sed -n '1,8p' || true
    fi
    exit 0
fi

# --- write modes need root -----------------------------------------------
if [[ $EUID -ne 0 ]]; then
    echo "This action edits $CONFIG_FILE and must run as root." >&2
    echo "Try: sudo bash $0 $mode" >&2
    exit 1
fi

# One-time backup.
if [[ ! -f "$BACKUP_FILE" ]]; then
    cp -p "$CONFIG_FILE" "$BACKUP_FILE"
    echo "Backed up original to: $BACKUP_FILE"
fi

# Build the desired block contents.
case "$mode" in
    imx296)
        block_body=$(cat <<'EOF'
# Mode: imx296 (Global Shutter Camera) - auto-detect handles it.
camera_auto_detect=1
#dtoverlay=imx500
EOF
)
        ;;
    imx500)
        block_body=$(cat <<'EOF'
# Mode: imx500 (Sony AI Camera) - Pi 4 needs explicit overlay.
camera_auto_detect=0
dtoverlay=imx500
EOF
)
        ;;
esac

# Strip any pre-existing managed block, plus stray top-level
# camera_auto_detect / dtoverlay=imx500 lines we'd otherwise duplicate.
tmp=$(mktemp)
awk -v b="$BLOCK_BEGIN" -v e="$BLOCK_END" '
    $0 ~ b {skip=1; next}
    $0 ~ e {skip=0; next}
    skip {next}
    /^[[:space:]]*camera_auto_detect[[:space:]]*=/ {next}
    /^[[:space:]]*dtoverlay[[:space:]]*=[[:space:]]*imx500[[:space:]]*$/ {next}
    {print}
' "$CONFIG_FILE" > "$tmp"

# Append the fresh managed block.
{
    # Ensure exactly one trailing newline before the block.
    if [[ -s "$tmp" ]] && [[ -n "$(tail -c1 "$tmp")" ]]; then echo; fi
    echo "$BLOCK_BEGIN"
    echo "$block_body"
    echo "$BLOCK_END"
} >> "$tmp"

# Replace atomically.
install -m 0755 -o root -g root "$tmp" "$CONFIG_FILE" 2>/dev/null || \
    cp "$tmp" "$CONFIG_FILE"
rm -f "$tmp"

echo "Camera mode set to: $mode"
echo
echo "Managed block now in $CONFIG_FILE:"
sed -n "/$BLOCK_BEGIN/,/$BLOCK_END/p" "$CONFIG_FILE"
echo
echo "Reboot required:  sudo reboot"

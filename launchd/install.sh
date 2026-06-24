#!/bin/bash
# install.sh — registers the daily pipeline as a macOS launchd job.
#
# Run this ONCE from the project root on the Mac that will run the
# pipeline daily:
#   bash launchd/install.sh
#
# It will:
#   1. Fill in the real project path inside the plist template
#   2. Copy it to ~/Library/LaunchAgents
#   3. Load it with launchctl
#
# To undo: bash launchd/uninstall.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLIST_NAME="com.nongnapat.loadforecasting.plist"
TEMPLATE_PATH="$SCRIPT_DIR/$PLIST_NAME"
DEST_PATH="$HOME/Library/LaunchAgents/$PLIST_NAME"

if [ ! -f "$TEMPLATE_PATH" ]; then
    echo "ERROR: template not found at $TEMPLATE_PATH"
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$PROJECT_DIR/output/logs"

# Substitute the placeholder with the real, absolute project path.
sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$TEMPLATE_PATH" > "$DEST_PATH"

chmod +x "$PROJECT_DIR/run_daily_mac.sh"

# Unload first in case a previous version is already loaded (ignore errors
# if it wasn't loaded yet).
launchctl unload "$DEST_PATH" 2>/dev/null || true
launchctl load "$DEST_PATH"

echo "Installed and loaded: $DEST_PATH"
echo "Project dir resolved to: $PROJECT_DIR"
echo ""
echo "The pipeline will run automatically every day at 06:00 (local time)."
echo "To change the time, edit $DEST_PATH directly (StartCalendarInterval),"
echo "then run: launchctl unload \"$DEST_PATH\" && launchctl load \"$DEST_PATH\""
echo ""
echo "To test it manually right now (without waiting for 06:00):"
echo "  launchctl start com.nongnapat.loadforecasting"
echo ""
echo "To check status:"
echo "  launchctl list | grep nongnapat"
echo ""
echo "Logs:"
echo "  Pipeline log:  $PROJECT_DIR/output/logs/pipeline.log"
echo "  Daily run log: $PROJECT_DIR/output/logs/daily_run.log"
echo "  launchd stdout: $PROJECT_DIR/output/logs/launchd_out.log"
echo "  launchd stderr: $PROJECT_DIR/output/logs/launchd_err.log"

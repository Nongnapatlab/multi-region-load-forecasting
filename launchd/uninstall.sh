#!/bin/bash
# uninstall.sh — unloads and removes the daily pipeline launchd job.
set -uo pipefail

PLIST_NAME="com.nongnapat.loadforecasting.plist"
DEST_PATH="$HOME/Library/LaunchAgents/$PLIST_NAME"

if [ -f "$DEST_PATH" ]; then
    launchctl unload "$DEST_PATH" 2>/dev/null || true
    rm "$DEST_PATH"
    echo "Removed: $DEST_PATH"
else
    echo "Nothing to remove: $DEST_PATH not found"
fi

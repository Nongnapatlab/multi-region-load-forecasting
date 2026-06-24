#!/bin/bash
# run_daily_mac.sh — daily pipeline runner for macOS (invoked by launchd)
#
# Usage (manual test): ./run_daily_mac.sh
# Usage (automatic): registered via the launchd plist, see launchd/README.md

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_DIR/output/logs"
DAILY_LOG="$LOG_DIR/daily_run.log"

mkdir -p "$LOG_DIR"

# Prefer a project virtualenv if present, otherwise fall back to whatever
# `python3` resolves to on PATH.
if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

{
    echo "============================================================"
    echo "Run started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Project dir: $PROJECT_DIR"
    echo "Python: $PYTHON_BIN"
    echo "============================================================"
} >> "$DAILY_LOG"

cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR" >> "$DAILY_LOG"; exit 1; }

"$PYTHON_BIN" src/main.py >> "$DAILY_LOG" 2>&1
EXIT_CODE=$?

{
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Run finished OK: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "Run FAILED (exit code $EXIT_CODE): $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    echo "============================================================"
} >> "$DAILY_LOG"

exit $EXIT_CODE

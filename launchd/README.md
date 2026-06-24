# Daily automatic run (macOS / launchd)

This sets up the pipeline to run automatically every morning on a Mac,
without needing to keep a terminal open.

## Files

- `run_daily_mac.sh` (project root) — the actual runner. Uses a project
  virtualenv at `.venv` if one exists, otherwise falls back to `python3`
  on PATH. Logs to `output/logs/daily_run.log`.
- `launchd/com.nongnapat.loadforecasting.plist` — template job
  definition. Contains `__PROJECT_DIR__` placeholders filled in
  automatically by `install.sh` — don't edit the placeholder by hand, it
  gets overwritten on every install.
- `launchd/install.sh` — one-time setup script.
- `launchd/uninstall.sh` — removes the scheduled job.

## One-time setup

From the project root, on the Mac that will run the pipeline daily:

```bash
bash launchd/install.sh
```

This fills in the real project path, copies the plist to
`~/Library/LaunchAgents/`, and loads it with `launchctl`.

## Default schedule

Runs every day at **06:00 local time**. To change it, edit the
`StartCalendarInterval` block in the *installed* copy:

```
~/Library/LaunchAgents/com.nongnapat.loadforecasting.plist
```

then reload:

```bash
launchctl unload ~/Library/LaunchAgents/com.nongnapat.loadforecasting.plist
launchctl load ~/Library/LaunchAgents/com.nongnapat.loadforecasting.plist
```

(Editing the template in the repo and re-running `install.sh` also works,
and is the way to make a permanent change that survives reinstalls.)

## Testing without waiting for 06:00

```bash
launchctl start com.nongnapat.loadforecasting
tail -f output/logs/daily_run.log
```

## Checking status

```bash
launchctl list | grep nongnapat
```

## Logs

- `output/logs/daily_run.log` — appended every run, one after another.
- `output/logs/pipeline.log` — detailed per-zone log from the most recent
  run only (overwritten each run by `main.py`'s own logging setup).
- `output/logs/launchd_out.log` / `launchd_err.log` — launchd-level
  stdout/stderr, useful if the job doesn't even reach
  `run_daily_mac.sh`'s own logging (e.g. script not found, not
  executable).

## Uninstalling

```bash
bash launchd/uninstall.sh
```

## Common issues

**"python3: command not found" in launchd_err.log but works fine in
Terminal.** launchd jobs run with a minimal PATH (no shell profile
sourcing). Set up a project virtualenv at `.venv` — `run_daily_mac.sh`
prefers it automatically when present:

```bash
cd /path/to/multi-region-load-forecasting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**`ModuleNotFoundError: No module named 'holidays'`.** Make sure
`requirements.txt` includes `holidays` (it should already, as of this
fix) and that you installed it into the same environment
`run_daily_mac.sh` actually uses — check which `PYTHON_BIN` it resolved
to in `daily_run.log`.

**Job doesn't seem to run at all.** Confirm it's loaded:
`launchctl list | grep nongnapat`. If not listed, re-run
`bash launchd/install.sh`.

**Permission denied on run_daily_mac.sh.** Run
`chmod +x run_daily_mac.sh` from the project root.

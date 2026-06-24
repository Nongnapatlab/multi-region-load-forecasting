from datetime import date, datetime
from pathlib import Path

import pandas as pd

from config import DATE_COL, ZERO_ACTUAL_THRESHOLD


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def export_today_predictions(predictions_df: pd.DataFrame, output_path: Path, run_date: date) -> pd.DataFrame:
    """Rows whose for_date falls on run_date (the day the pipeline is being
    run, not necessarily the latest date present in the data)."""
    dt = pd.to_datetime(predictions_df[DATE_COL])
    today_df = predictions_df[dt.dt.date == run_date]
    export_csv(today_df, output_path)
    return today_df


def export_latest_available_predictions(predictions_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Rows for the most recent for_date actually present in this run's
    predictions (per zone, since zones can have slightly different ranges)."""
    dt = pd.to_datetime(predictions_df[DATE_COL])
    df = predictions_df.copy()
    df["_date_only"] = dt.dt.date

    frames = []
    for zone, zone_df in df.groupby("zone"):
        latest_date = zone_df["_date_only"].max()
        frames.append(zone_df[zone_df["_date_only"] == latest_date])

    latest_df = pd.concat(frames, ignore_index=True).drop(columns=["_date_only"]) if frames else predictions_df.iloc[0:0]
    export_csv(latest_df, output_path)
    return latest_df


def export_future_predictions(predictions_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Rows whose actual is still a not-yet-realized placeholder (|actual| <
    ZERO_ACTUAL_THRESHOLD) — i.e. forecasts for dates the source server
    hasn't backfilled with real measurements yet."""
    future_df = predictions_df[predictions_df["actual"].abs() < ZERO_ACTUAL_THRESHOLD]
    export_csv(future_df, output_path)
    return future_df


def export_snapshot(predictions_df: pd.DataFrame, snapshots_dir: Path, run_timestamp: str) -> Path:
    """One immutable CSV per run, named with the run's timestamp, so past
    runs can always be inspected individually."""
    snapshot_path = snapshots_dir / f"predictions_{run_timestamp}.csv"
    export_csv(predictions_df, snapshot_path)
    return snapshot_path


def append_prediction_history(predictions_df: pd.DataFrame, history_path: Path, run_timestamp: str) -> pd.DataFrame:
    """Append this run's predictions to a cumulative history file, tagged
    with the run timestamp, so every prediction ever made for a given
    for_date/zone is preserved (not overwritten by later runs). This is what
    lets the dashboard later look up "what did we predict for today, back
    when we ran yesterday" (the day-ahead Plan value)."""
    tagged = predictions_df.copy()
    tagged["run_timestamp"] = run_timestamp
    tagged["run_date"] = datetime.strptime(run_timestamp, "%Y%m%d_%H%M%S").date()

    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        existing = pd.read_csv(history_path)
        combined = pd.concat([existing, tagged], ignore_index=True)
    else:
        combined = tagged

    combined.to_csv(history_path, index=False)
    return combined
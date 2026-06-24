"""Decision dashboard: one row per zone summarizing what an operator
actually needs to glance at each morning, similar in spirit to EGAT's
sysgen-style view.

Columns:
  - plan: the day-ahead forecast for *today*, i.e. whatever the ensemble
    predicted for today's date back when the pipeline last ran *before*
    today (typically yesterday morning). This is "load plan" in the EGAT
    sense — a forecast made in advance, not today's own forecast.
  - actual_yesterday_avg / actual_yesterday_max: realized demand yesterday.
  - peak_yesterday: yesterday's maximum actual demand, with the timestamp
    it occurred at.
  - today_forecast: today's ensemble forecast, made during *this* run.
  - plan_vs_actual_note: present only when both plan and actual_yesterday
    are available for the same date (i.e. a plan made 1+ runs ago for a day
    that has since become "yesterday"), comparing what was planned against
    what actually happened.

Note: plan and today_forecast will be numerically identical only if the
pipeline happens to run more than once for the same date; with a single
daily run they represent different forecast horizons by construction
(today's forecast vs. a forecast made on a prior run for a now-past date).
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import DATE_COL
from metrics import ZERO_ACTUAL_THRESHOLD


def _is_realized(actual: pd.Series) -> pd.Series:
    return actual.abs() >= ZERO_ACTUAL_THRESHOLD


def _load_history(history_path) -> pd.DataFrame | None:
    if not history_path.exists():
        return None
    history_df = pd.read_csv(history_path)
    if history_df.empty:
        return None
    history_df[DATE_COL] = pd.to_datetime(history_df[DATE_COL])
    history_df["run_date"] = pd.to_datetime(history_df["run_date"]).dt.date
    return history_df


def _plan_for_today(history_df: pd.DataFrame, zone: str, run_date: date) -> dict:
    """Find the most recent run strictly before run_date, and pull what
    that run predicted (ensemble) for today's date (run_date). Returns a
    dict with plan_total/plan_peak/plan_made_on, or all-None if no prior
    run made a forecast covering today."""
    zone_hist = history_df[(history_df["zone"] == zone) & (history_df["run_date"] < run_date)]
    if zone_hist.empty:
        return {"plan_total": None, "plan_peak": None, "plan_made_on_run_date": None}

    most_recent_prior_run = zone_hist["run_date"].max()
    prior_run_rows = zone_hist[zone_hist["run_date"] == most_recent_prior_run]

    rows_for_today = prior_run_rows[prior_run_rows[DATE_COL].dt.date == run_date]
    if rows_for_today.empty:
        return {"plan_total": None, "plan_peak": None, "plan_made_on_run_date": str(most_recent_prior_run)}

    return {
        "plan_total": round(float(rows_for_today["pred_ensemble"].sum()), 2),
        "plan_peak": round(float(rows_for_today["pred_ensemble"].max()), 2),
        "plan_made_on_run_date": str(most_recent_prior_run),
    }


def build_decision_dashboard(
    predictions_df: pd.DataFrame,
    history_path,
    run_date: date,
) -> pd.DataFrame:
    df = predictions_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    yesterday = run_date - timedelta(days=1)

    history_df = _load_history(history_path)

    rows = []
    for zone in sorted(df["zone"].unique()):
        zone_df = df[df["zone"] == zone]

        # --- Today forecast (this run's own ensemble output for run_date) ---
        today_rows = zone_df[zone_df[DATE_COL].dt.date == run_date]
        today_total = round(float(today_rows["pred_ensemble"].sum()), 2) if not today_rows.empty else None
        today_peak = round(float(today_rows["pred_ensemble"].max()), 2) if not today_rows.empty else None
        today_peak_time = None
        if not today_rows.empty and today_rows["pred_ensemble"].notna().any():
            today_peak_time = str(today_rows.loc[today_rows["pred_ensemble"].idxmax(), DATE_COL])

        # --- Actual yesterday (realized rows only) ---
        yesterday_rows = zone_df[zone_df[DATE_COL].dt.date == yesterday]
        yesterday_realized = yesterday_rows[_is_realized(yesterday_rows["actual"])]

        actual_yesterday_avg = None
        peak_yesterday = None
        peak_yesterday_time = None
        if not yesterday_realized.empty:
            actual_yesterday_avg = round(float(yesterday_realized["actual"].mean()), 2)
            peak_idx = yesterday_realized["actual"].idxmax()
            peak_yesterday = round(float(yesterday_realized.loc[peak_idx, "actual"]), 2)
            peak_yesterday_time = str(yesterday_realized.loc[peak_idx, DATE_COL])

        # --- Plan: day-ahead forecast for today, from a prior run ---
        plan = {"plan_total": None, "plan_peak": None, "plan_made_on_run_date": None}
        if history_df is not None:
            plan = _plan_for_today(history_df, zone, run_date)

        rows.append({
            "zone": zone,
            "date": str(run_date),
            "plan_total_mw": plan["plan_total"],
            "plan_peak_mw": plan["plan_peak"],
            "plan_made_on": plan["plan_made_on_run_date"],
            "actual_yesterday_avg_mw": actual_yesterday_avg,
            "peak_yesterday_mw": peak_yesterday,
            "peak_yesterday_time": peak_yesterday_time,
            "today_forecast_total_mw": today_total,
            "today_forecast_peak_mw": today_peak,
            "today_forecast_peak_time": today_peak_time,
        })

    return pd.DataFrame(rows)
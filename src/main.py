"""Entry point for the multi-region load forecasting pipeline.

Orchestrates:
1. Per-zone training & evaluation  (pipeline.py)
2. CSV exports                      (export_results.py)
3. Decision dashboard               (dashboard.py)
4. Excel report                     (report_excel.py)
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    CACHE_DIR,
    DIRECTORIES,
    LOGS_DIR,
    METRICS_DIR,
    OUTPUT_DIR,
    PREDICTIONS_DIR,
    ZONES,
)
from dashboard import build_decision_dashboard
from export_results import (
    append_prediction_history,
    export_csv,
    export_future_predictions,
    export_latest_available_predictions,
    export_snapshot,
    export_today_predictions,
)
from pipeline import (
    build_all_zone_summary,
    build_daily_mape,
    run_zone_pipeline,
)
from report_excel import build_excel_report


# ── Paths that only main.py needs to know about ───────────────────────────────
SNAPSHOTS_DIR   = OUTPUT_DIR / "snapshots"
HISTORY_PATH    = OUTPUT_DIR / "predictions" / "prediction_history.csv"
EXCEL_REPORT    = OUTPUT_DIR / "load_forecast_report.xlsx"


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def ensure_directories() -> None:
    for directory in DIRECTORIES:
        Path(directory).mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()
    setup_logging()

    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_date = datetime.now().date()
    logging.info("Pipeline started  (run_timestamp=%s)", run_ts)

    # ── 1. Per-zone pipeline ──────────────────────────────────────────────────
    all_predictions: list[pd.DataFrame] = []
    all_metrics:     list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []

    for zone_name, zone_config in ZONES.items():
        zone_predictions, zone_metrics, zone_diagnostics = run_zone_pipeline(
            zone_name, zone_config, CACHE_DIR
        )
        all_predictions.append(zone_predictions)
        all_metrics.append(zone_metrics)
        all_diagnostics.append(zone_diagnostics)

    predictions_df  = pd.concat(all_predictions,  ignore_index=True)
    metrics_df      = pd.concat(all_metrics,       ignore_index=True)
    diagnostics_df  = pd.concat(all_diagnostics,   ignore_index=True)

    summary_df      = build_all_zone_summary(metrics_df)
    metrics_full_df = pd.concat([metrics_df, summary_df], ignore_index=True)
    daily_mape_df   = build_daily_mape(predictions_df)

    # ── 2. CSV exports ────────────────────────────────────────────────────────
    export_csv(predictions_df,  PREDICTIONS_DIR / "all_zones_predictions.csv")
    export_csv(metrics_full_df, METRICS_DIR      / "all_zones_metrics.csv")
    export_csv(daily_mape_df,   METRICS_DIR      / "all_zones_daily_mape.csv")
    export_csv(diagnostics_df,  LOGS_DIR         / "diagnostics_summary.csv")

    today_df  = export_today_predictions(
        predictions_df,
        PREDICTIONS_DIR / "today_predictions.csv",
        run_date,
    )
    future_df = export_future_predictions(
        predictions_df,
        PREDICTIONS_DIR / "future_predictions.csv",
    )
    export_latest_available_predictions(
        predictions_df,
        PREDICTIONS_DIR / "latest_predictions.csv",
    )
    export_snapshot(predictions_df, SNAPSHOTS_DIR, run_ts)
    history_df = append_prediction_history(predictions_df, HISTORY_PATH, run_ts)

    logging.info("CSV exports complete")

    # ── 3. Decision dashboard ─────────────────────────────────────────────────
    dashboard_df = build_decision_dashboard(
        predictions_df=predictions_df,
        history_path=HISTORY_PATH,
        run_date=run_date,
    )
    export_csv(dashboard_df, OUTPUT_DIR / "decision_dashboard.csv")
    logging.info("Decision dashboard built (%d zones)", len(dashboard_df))

    # ── 4. Excel report ───────────────────────────────────────────────────────
    build_excel_report(
        output_path=EXCEL_REPORT,
        dashboard_df=dashboard_df,
        predictions_df=predictions_df,
        metrics_df=metrics_full_df,
        daily_mape_df=daily_mape_df,
        diagnostics_df=diagnostics_df,
        today_df=today_df if not today_df.empty else None,
        future_df=future_df if not future_df.empty else None,
    )
    logging.info("Excel report written → %s", EXCEL_REPORT)

    logging.info("Pipeline finished successfully (run_timestamp=%s)", run_ts)


if __name__ == "__main__":
    main()
    
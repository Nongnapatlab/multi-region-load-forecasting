import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    DATE_COL,
    TARGET_COL,
    CACHE_DIR,
    DIRECTORIES,
    ENSEMBLE_WEIGHTS,
    HISTORY_DIR,
    LOGS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    SNAPSHOTS_DIR,
    USE_LSTM,
    ZONES,
)
from dashboard import build_decision_dashboard
from data_loader import load_zone_data
from diagnostics import (
    build_bias_summary,
    build_diagnostics,
    build_error_by_hour,
    build_error_by_weekday,
    build_feature_importance,
)
from ensemble import weighted_average_ensemble
from export_results import (
    append_prediction_history,
    export_csv,
    export_future_predictions,
    export_latest_available_predictions,
    export_snapshot,
    export_today_predictions,
)
from feature_engineering import build_features
from metrics import add_absolute_percentage_error, evaluate_predictions
from preprocess import prepare_features
from report_excel import build_excel_report
from train_lgbm import train_lgbm_model
from train_lstm import predict_lstm, train_lstm_model
from train_xgb import train_xgb_model


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


def run_zone_pipeline(zone_name: str, zone_config: dict):
    logging.info("Starting zone: %s", zone_name)
    train_df, test_df = load_zone_data(zone_name, zone_config, CACHE_DIR)

    train_df = build_features(train_df)
    test_df = build_features(test_df)

    X_train, X_test, y_train, y_test, feature_cols = prepare_features(train_df, test_df)

    xgb_model = train_xgb_model(X_train, y_train)
    lgbm_model = train_lgbm_model(X_train, y_train)

    test_result = test_df[[DATE_COL, TARGET_COL, "zone"]].copy()
    test_result = test_result.rename(columns={TARGET_COL: "actual"})

    test_result["pred_xgb"] = xgb_model.predict(X_test)
    test_result["pred_lgbm"] = lgbm_model.predict(X_test)

    if USE_LSTM:
        lstm_model, x_scaler, y_scaler = train_lstm_model(X_train, y_train)
        test_result["pred_lstm"] = predict_lstm(lstm_model, x_scaler, y_scaler, X_test)
    else:
        test_result["pred_lstm"] = pd.NA

    test_result = weighted_average_ensemble(
        test_result,
        weights=ENSEMBLE_WEIGHTS,
        output_col="pred_ensemble",
    )

    for pred_col in ["pred_xgb", "pred_lgbm", "pred_lstm", "pred_ensemble"]:
        test_result = add_absolute_percentage_error(test_result, "actual", pred_col, f"ape_{pred_col}")

    metrics_rows = []
    for model_name, pred_col in [
        ("XGBoost", "pred_xgb"),
        ("LightGBM", "pred_lgbm"),
        ("LSTM", "pred_lstm"),
        ("Ensemble", "pred_ensemble"),
    ]:
        metric_values = evaluate_predictions(test_result["actual"], test_result[pred_col])
        metrics_rows.append({"zone": zone_name, "model": model_name, **metric_values})

    metrics_df = pd.DataFrame(metrics_rows)
    diagnostics_df = build_diagnostics(zone_name, train_df, test_df, feature_cols)

    pred_cols = ["pred_xgb", "pred_lgbm", "pred_lstm", "pred_ensemble"]
    importance_frames = [
        build_feature_importance(zone_name, "XGBoost", xgb_model, feature_cols),
        build_feature_importance(zone_name, "LightGBM", lgbm_model, feature_cols),
    ]
    importance_df = pd.concat([f for f in importance_frames if not f.empty], ignore_index=True) if any(not f.empty for f in importance_frames) else pd.DataFrame()
    error_by_hour_df = build_error_by_hour(zone_name, test_result, pred_cols)
    error_by_weekday_df = build_error_by_weekday(zone_name, test_result, pred_cols)

    logging.info("Finished zone: %s", zone_name)
    return test_result, metrics_df, diagnostics_df, importance_df, error_by_hour_df, error_by_weekday_df


def build_all_zone_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics across zones, weighted by how many rows actually
    contributed to each zone's score (n_rows_scored). A simple unweighted
    mean would treat a zone with 24 scored rows the same as one with 288."""
    rows = []
    for model_name in metrics_df["model"].dropna().unique():
        subset = metrics_df[metrics_df["model"] == model_name]
        weights = subset["n_rows_scored"].fillna(0)
        total_weight = weights.sum()

        def weighted_mean(col):
            if total_weight == 0:
                return float("nan")
            return (subset[col] * weights).sum() / total_weight

        rows.append(
            {
                "zone": "ALL",
                "model": model_name,
                "MAPE": weighted_mean("MAPE"),
                "MAE": weighted_mean("MAE"),
                "RMSE": weighted_mean("RMSE"),
                "BIAS": weighted_mean("BIAS"),
                "n_rows_total": subset["n_rows_total"].sum(),
                "n_rows_scored": subset["n_rows_scored"].sum(),
                "n_rows_excluded_zero_actual": subset["n_rows_excluded_zero_actual"].sum(),
            }
        )
    return pd.DataFrame(rows)


def build_daily_mape(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zone, zone_df in predictions_df.groupby("zone"):
        for model_name, ape_col in [
            ("XGBoost", "ape_pred_xgb"),
            ("LightGBM", "ape_pred_lgbm"),
            ("LSTM", "ape_pred_lstm"),
            ("Ensemble", "ape_pred_ensemble"),
        ]:
            if DATE_COL in zone_df.columns:
                grouped = zone_df.groupby(DATE_COL, dropna=False)[ape_col].mean().reset_index()
                grouped["zone"] = zone
                grouped["model"] = model_name
                grouped = grouped.rename(columns={ape_col: "daily_mape"})
                rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ensure_directories()
    setup_logging()

    run_started_at = datetime.now()
    run_timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    run_date = run_started_at.date()

    logging.info("Pipeline started | run_timestamp=%s | run_date=%s", run_timestamp, run_date)

    all_predictions = []
    all_metrics = []
    all_diagnostics = []
    all_importance = []
    all_error_by_hour = []
    all_error_by_weekday = []

    for zone_name, zone_config in ZONES.items():
        (
            zone_predictions,
            zone_metrics,
            zone_diagnostics,
            zone_importance,
            zone_error_by_hour,
            zone_error_by_weekday,
        ) = run_zone_pipeline(zone_name, zone_config)
        all_predictions.append(zone_predictions)
        all_metrics.append(zone_metrics)
        all_diagnostics.append(zone_diagnostics)
        if not zone_importance.empty:
            all_importance.append(zone_importance)
        if not zone_error_by_hour.empty:
            all_error_by_hour.append(zone_error_by_hour)
        if not zone_error_by_weekday.empty:
            all_error_by_weekday.append(zone_error_by_weekday)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    diagnostics_df = pd.concat(all_diagnostics, ignore_index=True)

    summary_df = build_all_zone_summary(metrics_df)
    metrics_full_df = pd.concat([metrics_df, summary_df], ignore_index=True)
    daily_mape_df = build_daily_mape(predictions_df)
    bias_summary_df = build_bias_summary(metrics_df)

    # --- Core outputs (overwritten each run: always reflect the latest run) ---
    export_csv(predictions_df, PREDICTIONS_DIR / "all_zones_predictions.csv")
    export_csv(metrics_full_df, METRICS_DIR / "all_zones_metrics.csv")
    export_csv(daily_mape_df, METRICS_DIR / "all_zones_daily_mape.csv")
    export_csv(diagnostics_df, LOGS_DIR / "diagnostics_summary.csv")
    export_csv(bias_summary_df, METRICS_DIR / "bias_summary.csv")

    if all_importance:
        export_csv(pd.concat(all_importance, ignore_index=True), LOGS_DIR / "feature_importance.csv")
    if all_error_by_hour:
        export_csv(pd.concat(all_error_by_hour, ignore_index=True), LOGS_DIR / "error_by_hour.csv")
    if all_error_by_weekday:
        export_csv(pd.concat(all_error_by_weekday, ignore_index=True), LOGS_DIR / "error_by_weekday.csv")

    # --- Filtered views of this run's predictions ---
    today_df = export_today_predictions(predictions_df, PREDICTIONS_DIR / "today_predictions.csv", run_date)
    export_latest_available_predictions(predictions_df, PREDICTIONS_DIR / "latest_available_predictions.csv")
    future_df = export_future_predictions(predictions_df, PREDICTIONS_DIR / "future_predictions.csv")

    # --- Append to cumulative history BEFORE building the dashboard, so the
    # dashboard's "plan" lookup logic stays based on *prior* runs only; this
    # run's own predictions are written to history but the dashboard reads
    # the file as it was before this line runs (predictions_df in memory is
    # used directly for "today_forecast", history file for "plan"). ---
    history_path = HISTORY_DIR / "prediction_history.csv"
    dashboard_df = build_decision_dashboard(predictions_df, history_path, run_date)
    append_prediction_history(predictions_df, history_path, run_timestamp)

    # --- Immutable per-run snapshot ---
    export_snapshot(predictions_df, SNAPSHOTS_DIR, run_timestamp)

    export_csv(dashboard_df, METRICS_DIR / "decision_dashboard.csv")

    # --- All-in-one Excel report ---
    build_excel_report(
        REPORTS_DIR / "all_in_one_forecasting_report.xlsx",
        dashboard_df=dashboard_df,
        predictions_df=predictions_df,
        metrics_df=metrics_full_df,
        daily_mape_df=daily_mape_df,
        diagnostics_df=diagnostics_df,
        bias_summary_df=bias_summary_df,
        today_df=today_df,
        future_df=future_df,
    )

    logging.info("Pipeline finished successfully | run_timestamp=%s", run_timestamp)


if __name__ == "__main__":
    main()
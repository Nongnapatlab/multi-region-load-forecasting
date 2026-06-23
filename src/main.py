import logging
from pathlib import Path

import pandas as pd

from config import (
    DATE_COL,
    TARGET_COL,
    CACHE_DIR,
    DIRECTORIES,
    LOGS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    USE_LSTM,
    ZONES,
)
from data_loader import load_zone_data
from diagnostics import build_diagnostics
from ensemble import weighted_average_ensemble
from export_results import export_csv
from feature_engineering import build_features
from metrics import add_absolute_percentage_error, evaluate_predictions
from preprocess import prepare_features
from train_lgbm import train_lgbm_model
from train_lstm import predict_lstm, train_lstm_model
from train_xgb import train_xgb_model


# Equal-weight ensemble:
# XGBoost + LightGBM + LSTM
# ถ้าอยากปรับน้ำหนักภายหลัง สามารถแก้เลขตรงนี้ได้
ENSEMBLE_WEIGHTS = {
    "pred_xgb": 1.0,
    "pred_lgbm": 1.0,
    "pred_lstm": 1.0,
}


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
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
        test_result = add_absolute_percentage_error(
            test_result,
            actual_col="actual",
            pred_col=pred_col,
            output_col=f"ape_{pred_col}",
        )

    metrics_rows = []

    for model_name, pred_col in [
        ("XGBoost", "pred_xgb"),
        ("LightGBM", "pred_lgbm"),
        ("LSTM", "pred_lstm"),
        ("Ensemble", "pred_ensemble"),
    ]:
        metric_values = evaluate_predictions(test_result["actual"], test_result[pred_col])
        metrics_rows.append(
            {
                "zone": zone_name,
                "model": model_name,
                **metric_values,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    diagnostics_df = build_diagnostics(zone_name, train_df, test_df, feature_cols)

    logging.info("Finished zone: %s", zone_name)

    return test_result, metrics_df, diagnostics_df


def build_all_zone_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    รวม metrics ทุกภาคเป็นแถว ALL

    ใช้ weighted average ตามจำนวนแถวที่ถูกนำมาคิดคะแนนจริง
    เพื่อไม่ให้ภาคที่มีข้อมูล scored น้อยมีน้ำหนักเท่าภาคที่มีข้อมูล scored มาก
    """
    rows = []

    for model_name in metrics_df["model"].dropna().unique():
        subset = metrics_df[metrics_df["model"] == model_name].copy()

        if "n_rows_scored" not in subset.columns:
            continue

        weights = subset["n_rows_scored"].fillna(0)
        total_weight = weights.sum()

        def weighted_mean(col: str):
            if total_weight == 0 or col not in subset.columns:
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
                "n_rows_total": subset["n_rows_total"].sum()
                if "n_rows_total" in subset.columns
                else float("nan"),
                "n_rows_scored": subset["n_rows_scored"].sum()
                if "n_rows_scored" in subset.columns
                else float("nan"),
                "n_rows_excluded_zero_actual": subset["n_rows_excluded_zero_actual"].sum()
                if "n_rows_excluded_zero_actual" in subset.columns
                else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def build_daily_mape(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง MAPE รายวันของแต่ละภาคและแต่ละโมเดล

    หมายเหตุ:
    - ถ้า DATE_COL เป็น timestamp รายชั่วโมง จะตัดเหลือแค่ date
    - ค่า APE ที่เป็น NaN จาก actual ใกล้ 0 จะถูก skip โดย mean()
    """
    if predictions_df.empty or DATE_COL not in predictions_df.columns:
        return pd.DataFrame()

    df = predictions_df.copy()
    df["date"] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.date

    rows = []

    for zone, zone_df in df.groupby("zone"):
        for model_name, ape_col in [
            ("XGBoost", "ape_pred_xgb"),
            ("LightGBM", "ape_pred_lgbm"),
            ("LSTM", "ape_pred_lstm"),
            ("Ensemble", "ape_pred_ensemble"),
        ]:
            if ape_col not in zone_df.columns:
                continue

            grouped = (
                zone_df.groupby("date", dropna=False)[ape_col]
                .mean()
                .reset_index()
                .rename(columns={ape_col: "daily_mape"})
            )

            grouped["zone"] = zone
            grouped["model"] = model_name

            grouped = grouped[["date", "zone", "model", "daily_mape"]]
            rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ensure_directories()
    setup_logging()

    logging.info("Pipeline started")

    all_predictions = []
    all_metrics = []
    all_diagnostics = []

    for zone_name, zone_config in ZONES.items():
        zone_predictions, zone_metrics, zone_diagnostics = run_zone_pipeline(
            zone_name,
            zone_config,
        )

        all_predictions.append(zone_predictions)
        all_metrics.append(zone_metrics)
        all_diagnostics.append(zone_diagnostics)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    diagnostics_df = pd.concat(all_diagnostics, ignore_index=True)

    summary_df = build_all_zone_summary(metrics_df)
    metrics_full_df = pd.concat([metrics_df, summary_df], ignore_index=True)

    daily_mape_df = build_daily_mape(predictions_df)

    export_csv(predictions_df, PREDICTIONS_DIR / "all_zones_predictions.csv")
    export_csv(metrics_full_df, METRICS_DIR / "all_zones_metrics.csv")
    export_csv(daily_mape_df, METRICS_DIR / "all_zones_daily_mape.csv")
    export_csv(diagnostics_df, LOGS_DIR / "diagnostics_summary.csv")

    logging.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
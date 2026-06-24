"""Zone-level training and evaluation pipeline.

Replaces the ad-hoc logic that was scattered inside main.py's
run_zone_pipeline() with a cleaner, model-agnostic structure that:
- Uses the new wrapper classes from models_ml and models_lstm
- Applies weighted ensemble (XGB + LGBM outperform LSTM significantly)
- Returns all artefacts needed by dashboard.py and report_excel.py
"""

import logging
import pandas as pd
import numpy as np

from config import DATE_COL, TARGET_COL, USE_LSTM
from data_loader import load_zone_data
from diagnostics import build_diagnostics
from ensemble import weighted_average_ensemble
from export_results import export_csv
from feature_engineering import build_features
from metrics import add_absolute_percentage_error, evaluate_predictions
from models_ml import XGBModel, LGBMModel
from models_lstm import LSTMModel
from preprocess import prepare_features

# Ensemble weights — LSTM is included but down-weighted given its higher MAPE.
# Adjust here without touching the rest of the code.
ENSEMBLE_WEIGHTS = {
    "pred_xgb": 0.40,
    "pred_lgbm": 0.40,
    "pred_lstm": 0.20,
}
ENSEMBLE_WEIGHTS_NO_LSTM = {
    "pred_xgb": 0.50,
    "pred_lgbm": 0.50,
}

MODEL_PRED_COLS = [
    ("XGBoost", "pred_xgb"),
    ("LightGBM", "pred_lgbm"),
    ("LSTM", "pred_lstm"),
    ("Ensemble_XGB_LGBM_LSTM", "pred_ensemble"),
    ("Ensemble_XGB_LGBM", "pred_ensemble_xgb_lgbm"),
]


def _build_metrics_rows(zone_name: str, test_result: pd.DataFrame) -> list[dict]:
    rows = []
    for model_name, pred_col in MODEL_PRED_COLS:
        if pred_col not in test_result.columns:
            continue
        metric_values = evaluate_predictions(
            test_result["actual"], test_result[pred_col]
        )
        rows.append({"zone": zone_name, "model": model_name, **metric_values})
    return rows


def run_zone_pipeline(
    zone_name: str,
    zone_config: dict,
    cache_dir,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train all models for one zone, return (predictions, metrics, diagnostics)."""

    logging.info("Starting zone: %s", zone_name)

    # ── Data ──────────────────────────────────────────────────────────
    train_df, test_df = load_zone_data(zone_name, zone_config, cache_dir)
    train_df = build_features(train_df)
    test_df  = build_features(test_df)

    X_train, X_test, y_train, y_test, feature_cols = prepare_features(
        train_df, test_df
    )

    # ── Train ─────────────────────────────────────────────────────────
    xgb_model  = XGBModel().fit(X_train, y_train)
    lgbm_model = LGBMModel().fit(X_train, y_train)

    # ── Predictions frame ─────────────────────────────────────────────
    test_result = test_df[[DATE_COL, TARGET_COL, "zone"]].copy()
    test_result = test_result.rename(columns={TARGET_COL: "actual"})
    test_result["pred_xgb"]  = xgb_model.predict(X_test)
    test_result["pred_lgbm"] = lgbm_model.predict(X_test)

    # Ensemble of XGBoost + LightGBM only, computed regardless of whether
    # LSTM is used. This gives a stable baseline to compare the full
    # 3-model ensemble against — useful since LSTM's scoreable rows are
    # often very few (sequence warm-up consumes most of a short test set),
    # so it's hard to tell from pred_ensemble alone whether LSTM is helping
    # or hurting.
    test_result = weighted_average_ensemble(
        test_result,
        weights=ENSEMBLE_WEIGHTS_NO_LSTM,
        output_col="pred_ensemble_xgb_lgbm",
    )

    if USE_LSTM:
        lstm_model = LSTMModel().fit(X_train, y_train)
        test_result["pred_lstm"] = lstm_model.predict(X_test)
        weights = ENSEMBLE_WEIGHTS
    else:
        test_result["pred_lstm"] = pd.NA
        weights = ENSEMBLE_WEIGHTS_NO_LSTM

    test_result = weighted_average_ensemble(
        test_result,
        weights={k: v for k, v in weights.items() if k in test_result.columns},
        output_col="pred_ensemble",
    )

    # ── Per-row APE ───────────────────────────────────────────────────
    for _, pred_col in MODEL_PRED_COLS:
        if pred_col in test_result.columns:
            test_result = add_absolute_percentage_error(
                test_result, "actual", pred_col, f"ape_{pred_col}"
            )

    # ── Zone metrics ──────────────────────────────────────────────────
    metrics_rows = _build_metrics_rows(zone_name, test_result)
    metrics_df   = pd.DataFrame(metrics_rows)

    # ── Diagnostics ───────────────────────────────────────────────────
    diagnostics_df = build_diagnostics(
        zone_name, train_df, test_df, feature_cols
    )

    logging.info("Finished zone: %s", zone_name)
    return test_result, metrics_df, diagnostics_df


def build_all_zone_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Append an 'ALL' row per model, weighted by n_rows_scored so a zone
    that only had 1 scoreable row (e.g. LSTM during its sequence warm-up)
    doesn't count the same as a zone with 24 scoreable rows."""
    rows = []
    for model_name in metrics_df["model"].dropna().unique():
        subset = metrics_df[metrics_df["model"] == model_name]
        weights = subset["n_rows_scored"].fillna(0) if "n_rows_scored" in subset.columns else pd.Series([1] * len(subset))
        total_weight = weights.sum()

        def weighted_mean(col):
            if total_weight == 0:
                return float("nan")
            return (subset[col] * weights).sum() / total_weight

        row = {
            "zone": "ALL",
            "model": model_name,
            "MAPE": weighted_mean("MAPE"),
            "MAE": weighted_mean("MAE"),
            "RMSE": weighted_mean("RMSE"),
            "BIAS": weighted_mean("BIAS"),
        }
        if "n_rows_scored" in subset.columns:
            row["n_rows_total"] = subset["n_rows_total"].sum()
            row["n_rows_scored"] = subset["n_rows_scored"].sum()
            row["n_rows_excluded_zero_actual"] = subset["n_rows_excluded_zero_actual"].sum()
        rows.append(row)
    return pd.DataFrame(rows)


def build_daily_mape(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zone, zone_df in predictions_df.groupby("zone"):
        for model_name, ape_col in [
            ("XGBoost", "ape_pred_xgb"),
            ("LightGBM", "ape_pred_lgbm"),
            ("LSTM", "ape_pred_lstm"),
            ("Ensemble_XGB_LGBM_LSTM", "ape_pred_ensemble"),
            ("Ensemble_XGB_LGBM", "ape_pred_ensemble_xgb_lgbm"),
        ]:
            if DATE_COL not in zone_df.columns or ape_col not in zone_df.columns:
                continue
            grouped = (
                zone_df.groupby(DATE_COL, dropna=False)[ape_col]
                .mean()
                .reset_index()
            )
            grouped["zone"]  = zone
            grouped["model"] = model_name
            grouped = grouped.rename(columns={ape_col: "daily_mape"})
            rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
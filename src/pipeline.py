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
from diagnostics import build_diagnostics, build_distribution_shift_check, build_feature_importance
from ensemble import weighted_average_ensemble
from export_results import export_csv
from feature_engineering import build_features
from metrics import add_absolute_percentage_error, evaluate_predictions
from models_ml import XGBModel, LGBMModel
from models_lstm import LSTMModel
from preprocess import prepare_features
from typing import Optional

# Ensemble weights — LSTM is included but down-weighted given its higher MAPE.
# Adjust here without touching the rest of the code.
ENSEMBLE_WEIGHTS = {
    "pred_xgb":  0.50,
    "pred_lgbm": 0.50,
    "pred_lstm": 0.00,
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
    ensemble_weights: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train all models for one zone, return (predictions, metrics,
    diagnostics, feature_importance, distribution_shift).

    ensemble_weights:
        ถ้าส่งมา → ใช้ weights นั้น (โหลดมาจาก optimal_weights.json ใน main.py)
        ถ้าไม่ส่ง → ใช้ ENSEMBLE_WEIGHTS จาก pipeline.py (hardcoded default)
        เหตุผลที่รับ param นี้จากข้างนอก: ให้ main.py โหลด optimal weights ก่อน
        แล้วส่งเข้ามา — ทำให้ทุก zone ใช้ weights เดียวกัน และ testable แยกได้
    """

    logging.info("Starting zone: %s", zone_name)

    # ── Data ──────────────────────────────────────────────────────────
    train_df, test_df = load_zone_data(zone_name, zone_config, cache_dir)

    # ── บันทึก actual จริงก่อน build_features จะ ffill requirement ──────
    # _fill_placeholder_zeros ใน feature_engineering.py ffill requirement
    # เพื่อให้ lag features ได้ค่าที่สมเหตุสมผลแทน 0 — ถูกต้องสำหรับ feature
    # แต่ถ้าปล่อยให้ test_df[TARGET_COL] ถูก ffill แล้วเอามาตั้งเป็น actual
    # actual จะกลายเป็นค่าซ้ำ ๆ (flat line) ซึ่งทำให้ MAPE ผิดทั้งหมด
    # แก้ไข: เก็บ original targets ไว้ก่อน แล้วค่อยเอามาใส่หลัง build_features
    original_test_targets = test_df[TARGET_COL].copy()

    train_df = build_features(train_df)
    test_df  = build_features(test_df)

    X_train, X_test, y_train, y_test, feature_cols = prepare_features(
        train_df, test_df
    )

    # ── Train ─────────────────────────────────────────────────────────
    xgb_model  = XGBModel().fit(X_train, y_train)
    lgbm_model = LGBMModel().fit(X_train, y_train)

    # ── Predictions frame (ใช้ original targets ไม่ใช่ ffill'd version) ──
    test_result = test_df[[DATE_COL, "zone"]].copy()
    test_result["actual"] = original_test_targets.values  # ← original เท่านั้น
    test_result["pred_xgb"]  = xgb_model.predict(X_test)
    test_result["pred_lgbm"] = lgbm_model.predict(X_test)

    # ── Resolve weights: ใช้ที่ส่งมา ถ้าไม่ส่งมาใช้ default ────────────────
    active_weights = ensemble_weights if ensemble_weights is not None else ENSEMBLE_WEIGHTS

    # Ensemble XGB+LGBM เสมอ (baseline ไม่ขึ้นกับ LSTM)
    test_result = weighted_average_ensemble(
        test_result,
        weights=ENSEMBLE_WEIGHTS_NO_LSTM,
        output_col="pred_ensemble_xgb_lgbm",
    )

    if USE_LSTM:
        lstm_model = LSTMModel().fit(X_train, y_train)
        test_result["pred_lstm"] = lstm_model.predict(X_test)
        weights = active_weights       # ← ใช้ optimal weights ที่รับมา
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
    distribution_shift_df = build_distribution_shift_check(zone_name, train_df, test_df)

    importance_frames = []
    for model_name, model in [("XGBoost", xgb_model), ("LightGBM", lgbm_model)]:
        fi_df = build_feature_importance(zone_name, model, feature_cols, model_name)
        if not fi_df.empty:
            importance_frames.append(fi_df)
    feature_importance_df = (
        pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    )

    logging.info("Finished zone: %s", zone_name)
    return test_result, metrics_df, diagnostics_df, feature_importance_df, distribution_shift_df


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
    """
    MAPE รายวัน (ค่าเฉลี่ย APE ของทุก time slot ใน 1 วัน)

    บั๊กเดิม:
        groupby(DATE_COL) → DATE_COL มี timestamp เต็ม (เช่น "2026-06-24 00:30:00")
        ทำให้แต่ละ timestamp กลายเป็น group ของตัวเอง = ไม่มีการ aggregate จริง

    แก้ไข:
        1. แยก date_only (ไม่มีเวลา) แล้ว groupby date_only
           → ได้ค่าเฉลี่ย APE ของทุก time slot ในวันนั้นจริง ๆ
        2. กรองเฉพาะ scored rows (actual > threshold) ก่อน groupby
           → future rows ที่ actual=0 จะทำให้ APE = inf ซึ่งจะปน mean ไป
           → daily_mape จะมีเฉพาะวันที่มี actual จริงเท่านั้น (แปลผลได้ถูกต้อง)
    """
    from metrics import ZERO_ACTUAL_THRESHOLD

    rows = []
    for zone, zone_df in predictions_df.groupby("zone"):
        # ─── แก้ไข: ใช้ date_only แทน full timestamp ───────────────────────
        zone_df = zone_df.copy()
        zone_df["_date"] = pd.to_datetime(zone_df[DATE_COL], errors="coerce").dt.date

        # ─── แก้ไข: กรองเฉพาะ scored rows ────────────────────────────────
        scored_df = zone_df[zone_df["actual"].abs() >= ZERO_ACTUAL_THRESHOLD]
        if scored_df.empty:
            continue

        for model_name, ape_col in [
            ("XGBoost",             "ape_pred_xgb"),
            ("LightGBM",            "ape_pred_lgbm"),
            ("LSTM",                "ape_pred_lstm"),
            ("Ensemble_XGB_LGBM_LSTM", "ape_pred_ensemble"),
            ("Ensemble_XGB_LGBM",   "ape_pred_ensemble_xgb_lgbm"),
        ]:
            if ape_col not in scored_df.columns:
                continue
            grouped = (
                scored_df.groupby("_date", dropna=False)[ape_col]
                .mean()
                .reset_index()
                .rename(columns={"_date": "date", ape_col: "daily_mape"})
            )
            grouped["zone"]  = zone
            grouped["model"] = model_name
            rows.append(grouped[["date", "zone", "model", "daily_mape"]])

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
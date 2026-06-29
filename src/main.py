"""
main.py — Entry point สำหรับ multi-region load forecasting pipeline

Flow ทั้งหมด:
  1. โหลด optimal ensemble weights จาก disk (ถ้ามี)   ← ใหม่
  2. Run per-zone pipeline (train + predict)
  3. Auto-adaptation:                                   ← ใหม่
     a) check_drift  → flag ภาคที่ distribution เปลี่ยน
     b) check_mape_alerts → flag ภาคที่ MAPE เกิน threshold
     c) compute & apply bias corrections → แก้ future predictions
     d) optimise_ensemble_weights → บันทึก weights ที่ดีขึ้นสำหรับ run ถัดไป
  4. Export CSV
  5. Build decision dashboard
  6. Build Excel report

ทำไม optimal weights ถูกโหลดก่อน run แต่ optimise หลัง run?
  - โหลดก่อน:  ใช้ weights ที่ดีที่สุดที่รู้ณ ตอนนี้ (จาก run ก่อน ๆ)
  - optimise หลัง: รวม scored rows ของ run นี้เข้าไปด้วย แล้วค่อยอัปเดต
                   ผลจะใช้ได้ใน run ถัดไป ไม่ใช่ run นี้ (เพื่อไม่ให้ look-ahead)
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from auto_adapt import (
    apply_bias_corrections,
    check_drift,
    check_mape_alerts,
    compute_bias_corrections,
    load_optimal_weights,
    optimise_ensemble_weights,
)
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
    ENSEMBLE_WEIGHTS,          # default weights (fallback)
    build_all_zone_summary,
    build_daily_mape,
    run_zone_pipeline,
)
from report_excel import build_excel_report

SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"
HISTORY_PATH  = OUTPUT_DIR / "predictions" / "prediction_history.csv"
EXCEL_REPORT  = OUTPUT_DIR / "load_forecast_report.xlsx"


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
    for d in DIRECTORIES:
        Path(d).mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()
    setup_logging()

    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_date = datetime.now().date()
    logging.info("═" * 60)
    logging.info("Pipeline started  (run_timestamp=%s)", run_ts)

    # ── 1. โหลด optimal weights ─────────────────────────────────────────────
    # ถ้ายังไม่มีไฟล์ (run แรก) จะใช้ ENSEMBLE_WEIGHTS จาก pipeline.py
    # ถ้ามีไฟล์แล้ว → ใช้ weights ที่ optimize ไว้จาก run ก่อน ๆ
    active_weights = load_optimal_weights(fallback=ENSEMBLE_WEIGHTS)

    # ── 2. Per-zone pipeline ─────────────────────────────────────────────────
    all_predictions: list[pd.DataFrame] = []
    all_metrics:     list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    all_importance:  list[pd.DataFrame] = []
    all_dist_shift:  list[pd.DataFrame] = []

    for zone_name, zone_config in ZONES.items():
        zone_preds, zone_metrics, zone_diag, zone_imp, zone_shift = run_zone_pipeline(
            zone_name, zone_config, CACHE_DIR,
            ensemble_weights=active_weights,   # ← ส่ง optimal weights เข้าไป
        )
        all_predictions.append(zone_preds)
        all_metrics.append(zone_metrics)
        all_diagnostics.append(zone_diag)
        if not zone_imp.empty:    all_importance.append(zone_imp)
        if not zone_shift.empty:  all_dist_shift.append(zone_shift)

    predictions_df = pd.concat(all_predictions,  ignore_index=True)
    metrics_df     = pd.concat(all_metrics,      ignore_index=True)
    diagnostics_df = pd.concat(all_diagnostics,  ignore_index=True)
    importance_df  = pd.concat(all_importance,   ignore_index=True) if all_importance  else pd.DataFrame()
    dist_shift_df  = pd.concat(all_dist_shift,   ignore_index=True) if all_dist_shift  else pd.DataFrame()

    summary_df      = build_all_zone_summary(metrics_df)
    metrics_full_df = pd.concat([metrics_df, summary_df], ignore_index=True)

    # ── 3. Auto-adaptation ───────────────────────────────────────────────────
    logging.info("─" * 40)
    logging.info("Auto-adaptation layer")

    # 3a. Drift detection — อ่าน z-score จาก distribution_shift
    #     flagged_zones = {zone: z_score} เฉพาะภาคที่เกิน DRIFT_Z_RETRAIN
    flagged_zones = check_drift(dist_shift_df)

    # 3b. MAPE alerts — flag ภาคที่ ensemble MAPE เกิน threshold
    mape_alerts = check_mape_alerts(metrics_full_df)
    if mape_alerts:
        logging.critical("MAPE ALERTS: %d ภาคที่ต้องการตรวจสอบ", len(mape_alerts))

    # 3c. Bias correction — ใช้ scored history ทั้งหมด ไม่ใช่แค่ drift zones
    #     เหตุผล: แม้ z-score ยังไม่ถึง threshold แต่ถ้า bias สะสมเกิน
    #     MIN_SCORED_FOR_BIAS_CORRECTION แถว ก็ควรชดเชยแล้ว
    corrections = compute_bias_corrections(HISTORY_PATH)
    if corrections:
        predictions_df = apply_bias_corrections(predictions_df, corrections)
        logging.info(
            "Bias corrections applied: %s",
            {z: f"{c:+.1f} MW" for z, c in corrections.items()},
        )
    else:
        logging.info("Bias correction: ยังไม่มีข้อมูลพอ หรือ disabled")

    # 3d. Optimise weights (สำหรับ run ถัดไป ไม่ใช่ run นี้)
    #     ถ้า history มีข้อมูลพอ → อัปเดต optimal_weights.json
    new_weights = optimise_ensemble_weights(HISTORY_PATH, fallback=active_weights)
    if new_weights:
        logging.info(
            "Optimal weights updated for next run: XGB=%.1f%% LGBM=%.1f%% LSTM=%.1f%%",
            new_weights["pred_xgb"]  * 100,
            new_weights["pred_lgbm"] * 100,
            new_weights["pred_lstm"] * 100,
        )
    else:
        logging.info("Weight optimisation: ยังไม่มีข้อมูลพอ หรือ disabled")

    logging.info("─" * 40)

    # ── 4. daily_mape (คำนวณหลัง bias correction เพื่อให้ตรงกับ prediction จริง)
    daily_mape_df = build_daily_mape(predictions_df)

    # ── 5. CSV exports ───────────────────────────────────────────────────────
    export_csv(predictions_df,  PREDICTIONS_DIR / "all_zones_predictions.csv")
    export_csv(metrics_full_df, METRICS_DIR     / "all_zones_metrics.csv")
    export_csv(daily_mape_df,   METRICS_DIR     / "all_zones_daily_mape.csv")
    export_csv(diagnostics_df,  LOGS_DIR        / "diagnostics_summary.csv")

    if not importance_df.empty:
        export_csv(importance_df, LOGS_DIR / "feature_importance.csv")
    if not dist_shift_df.empty:
        export_csv(dist_shift_df, LOGS_DIR / "distribution_shift_check.csv")

    today_df  = export_today_predictions(predictions_df,  PREDICTIONS_DIR / "today_predictions.csv",  run_date)
    future_df = export_future_predictions(predictions_df, PREDICTIONS_DIR / "future_predictions.csv")
    export_latest_available_predictions(predictions_df,   PREDICTIONS_DIR / "latest_predictions.csv")
    export_snapshot(predictions_df, SNAPSHOTS_DIR, run_ts)

    # append ลง history (รวมถึงค่าที่ bias-corrected แล้วสำหรับ future rows)
    append_prediction_history(predictions_df, HISTORY_PATH, run_ts)
    logging.info("CSV exports complete")

    # ── 6. Decision dashboard ────────────────────────────────────────────────
    dashboard_df = build_decision_dashboard(
        predictions_df=predictions_df,
        history_path=HISTORY_PATH,
        run_date=run_date,
    )
    export_csv(dashboard_df, OUTPUT_DIR / "decision_dashboard.csv")
    logging.info("Decision dashboard built (%d zones)", len(dashboard_df))

    # ── 7. Excel report ──────────────────────────────────────────────────────
    build_excel_report(
        output_path=EXCEL_REPORT,
        dashboard_df=dashboard_df,
        predictions_df=predictions_df,
        metrics_df=metrics_full_df,
        daily_mape_df=daily_mape_df,
        diagnostics_df=diagnostics_df,
        today_df=today_df        if not today_df.empty  else None,
        future_df=future_df      if not future_df.empty else None,
        feature_importance_df=importance_df if not importance_df.empty else None,
        distribution_shift_df=dist_shift_df if not dist_shift_df.empty else None,
    )
    logging.info("Excel report → %s", EXCEL_REPORT)
    logging.info("Pipeline finished successfully (run_timestamp=%s)", run_ts)
    logging.info("═" * 60)


if __name__ == "__main__":
    main()
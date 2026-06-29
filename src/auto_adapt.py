"""
auto_adapt.py
─────────────────────────────────────────────────────────────────────────────
ชั้น Automatic Adaptation — รันหลัง pipeline แต่ละครั้ง

ทำไมต้องมีไฟล์นี้?
  ถ้าไม่มี: ทุกครั้งที่ข้อมูลเปลี่ยน (demand spike, drift ตามฤดูกาล)
  ต้องมาแก้โค้ด หรือปรับน้ำหนักด้วยมือ = ไม่ scale ได้
  ถ้ามี:    ระบบตรวจ → วิเคราะห์ → ปรับ → บันทึก → ใช้ run ถัดไป
            โดย operator แค่เปลี่ยน threshold ใน config.py

กลไก 4 อย่าง (ควบคุมทั้งหมดจาก config.py):
─────────────────────────────────────────────────────────────────────────────
1. check_drift()
   อ่านผล distribution_shift จาก diagnostics.py
   ถ้า z-score > DRIFT_Z_WARN  → log WARNING
   ถ้า z-score > DRIFT_Z_RETRAIN → log CRITICAL + flag ให้ bias correction ทำงาน

2. compute_bias_corrections()
   ดึง scored rows จาก prediction_history.csv
   คำนวณ mean(pred_ensemble - actual) = systematic bias per zone
   คืน dict {zone: correction_mw} ที่จะบวกเพิ่มเข้าไปใน future predictions

3. apply_bias_corrections()
   บวก correction เฉพาะ future rows (actual ≈ 0) เท่านั้น
   เหตุผล: scored rows มี actual จริงแล้ว ไม่ควรแก้ retroactively
           future rows คือสิ่งที่จะ "plan" จริง ๆ — ต้องแม่นที่สุด

4. optimise_ensemble_weights()
   ใช้ scipy.optimize.minimize หาน้ำหนัก [w_xgb, w_lgbm, w_lstm] ที่
   minimize MAPE บน history จริง (constraint: sum=1, แต่ละตัว >= 0)
   บันทึกลง output/weights/optimal_weights.json
   run ถัดไปจะโหลดมาใช้อัตโนมัติผ่าน load_optimal_weights()

ทุก event ถูก append ลง output/logs/adaptation_log.csv
เพื่อ traceability ว่า "run ไหน แก้อะไร เพราะอะไร"
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    BIAS_CORRECTION_ENABLED,
    DRIFT_Z_RETRAIN,
    DRIFT_Z_WARN,
    LOGS_DIR,
    MAPE_ALERT_PCT,
    MIN_SCORED_FOR_BIAS_CORRECTION,
    MIN_SCORED_FOR_WEIGHT_OPT,
    WEIGHT_OPT_ENABLED,
    WEIGHTS_DIR,
)

ADAPT_LOG_PATH       = LOGS_DIR / "adaptation_log.csv"
OPTIMAL_WEIGHTS_PATH = WEIGHTS_DIR / "optimal_weights.json"
ZERO_ACTUAL_THRESHOLD = 1.0   # MW — ต่ำกว่านี้ถือว่าเป็น placeholder ยังไม่มีค่าจริง


# ══════════════════════════════════════════════════════════════════════════════
# Utility: event logger
# ══════════════════════════════════════════════════════════════════════════════

def _log_event(event_type: str, zone: str, detail: str, value: Optional[float] = None) -> None:
    """Append 1 row ลง adaptation_log.csv — ไม่ลบประวัติเก่า"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "event_type": event_type,
        "zone":       zone,
        "detail":     detail,
        "value":      value,
    }])
    write_header = not ADAPT_LOG_PATH.exists()
    row.to_csv(ADAPT_LOG_PATH, mode="a", header=write_header, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Distribution drift detection
# ══════════════════════════════════════════════════════════════════════════════

def check_drift(dist_shift_df: pd.DataFrame) -> dict[str, float]:
    """
    อ่านผล distribution_shift และ log/flag ภาคที่มี z-score เกิน threshold

    Returns
    -------
    dict {zone: z_score} เฉพาะภาคที่เกิน DRIFT_Z_RETRAIN เท่านั้น
    (ใช้ใน main.py เพื่อตัดสินใจว่าจะ trigger bias correction ไหม)
    """
    if dist_shift_df.empty:
        return {}

    # pivot ให้ index=zone, columns=metric
    pivot = dist_shift_df.pivot_table(
        index="zone", columns="metric", values="value", aggfunc="first"
    )

    flagged: dict[str, float] = {}

    for zone in pivot.index:
        z = pivot.loc[zone].get("z_score_vs_train", np.nan)
        if pd.isna(z):
            continue

        abs_z = abs(float(z))

        if abs_z >= DRIFT_Z_RETRAIN:
            # ร้ายแรง — ข้อมูล test อยู่ไกลจาก train มากเกิน 2.5 SD
            # โมเดลพยากรณ์จาก distribution ที่ไม่ match → systematic bias สูง
            logging.critical(
                "DRIFT CRITICAL | %-3s | z=%.2f ≥ %.1f | "
                "bias correction จะ run อัตโนมัติ run นี้",
                zone, z, DRIFT_Z_RETRAIN,
            )
            _log_event("DRIFT_CRITICAL", zone, f"z={z:.3f}", float(z))
            flagged[zone] = float(z)

        elif abs_z >= DRIFT_Z_WARN:
            # เฝ้าระวัง — ยังอยู่ในขอบเขตปกติ แต่ trend เริ่มเปลี่ยน
            logging.warning(
                "DRIFT WARNING  | %-3s | z=%.2f ≥ %.1f | "
                "เฝ้าดู — ยังไม่ trigger correction",
                zone, z, DRIFT_Z_WARN,
            )
            _log_event("DRIFT_WARNING", zone, f"z={z:.3f}", float(z))

    return flagged   # {zone: z} เฉพาะ critical เท่านั้น


# ══════════════════════════════════════════════════════════════════════════════
# 2. Bias correction
# ══════════════════════════════════════════════════════════════════════════════

def _load_scored_history(history_path: Path) -> pd.DataFrame:
    """โหลด prediction_history เฉพาะแถวที่มี actual จริง (> threshold)"""
    if not history_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(history_path)
    if "actual" not in df.columns:
        return pd.DataFrame()
    return df[df["actual"].abs() >= ZERO_ACTUAL_THRESHOLD].copy()


def compute_bias_corrections(history_path: Path) -> dict[str, float]:
    """
    คำนวณ systematic bias ต่อภาคจาก scored history

    Logic:
        bias        = mean(pred_ensemble - actual)   ← ถ้าลบ = under-predict
        correction  = -bias                          ← บวกเพื่อชดเชย

    ตัวอย่าง:
        NEC bias = -163 MW  →  correction = +163 MW
        → future predictions ของ NEC จะถูกบวก 163 MW เพื่อชดเชย

    จะ return {} ถ้า:
        - BIAS_CORRECTION_ENABLED = False ใน config
        - มี scored rows น้อยกว่า MIN_SCORED_FOR_BIAS_CORRECTION
          (bias estimate ไม่น่าเชื่อถือถ้ามีข้อมูลน้อยเกินไป)
    """
    if not BIAS_CORRECTION_ENABLED:
        logging.info("Bias correction disabled (BIAS_CORRECTION_ENABLED=False)")
        return {}

    scored = _load_scored_history(history_path)
    if scored.empty or "pred_ensemble" not in scored.columns:
        logging.info("Bias correction skipped — history ว่างหรือยังไม่มี pred_ensemble")
        return {}

    corrections: dict[str, float] = {}

    for zone, g in scored.groupby("zone"):
        n = len(g)
        if n < MIN_SCORED_FOR_BIAS_CORRECTION:
            logging.info(
                "Bias correction skipped | %-3s | %d scored rows (ต้องการ ≥ %d)",
                zone, n, MIN_SCORED_FOR_BIAS_CORRECTION,
            )
            continue

        bias       = float((g["pred_ensemble"] - g["actual"]).mean())
        correction = -bias
        direction  = "UNDER" if bias < 0 else "OVER"

        logging.info(
            "Bias correction | %-3s | %s-predict %.1f MW avg จาก %d rows "
            "→ +%.1f MW จะถูกบวกใน future predictions",
            zone, direction, abs(bias), n, correction,
        )
        _log_event("BIAS_CORRECTION_COMPUTED", str(zone),
                   f"bias={bias:.2f} corr={correction:.2f} n={n}", correction)
        corrections[str(zone)] = correction

    return corrections


def apply_bias_corrections(
    predictions_df: pd.DataFrame,
    corrections: dict[str, float],
) -> pd.DataFrame:
    """
    บวก correction ลง prediction columns เฉพาะ future rows (actual ≈ 0)

    ทำไมแก้แค่ future rows?
        scored rows (actual > 0): มีค่าจริงแล้ว — MAPE ที่คำนวณไว้ถูกต้อง
                                   การย้อนกลับไปแก้ไม่มีประโยชน์และทำให้สับสน
        future rows (actual = 0): คือสิ่งที่ส่งให้ EGAT เป็น "Plan" จริง ๆ
                                   ต้องแม่นที่สุด → ชดเชย systematic error ก่อน export
    """
    if not corrections:
        return predictions_df

    pred_cols  = [c for c in predictions_df.columns if c.startswith("pred_")]
    df         = predictions_df.copy()
    future_mask = df["actual"].abs() < ZERO_ACTUAL_THRESHOLD

    for zone, corr in corrections.items():
        mask = (df["zone"] == zone) & future_mask
        if not mask.any():
            continue
        for col in pred_cols:
            if col in df.columns:
                df.loc[mask, col] = (df.loc[mask, col] + corr).clip(lower=0)
        logging.debug("Applied %.1f MW correction to %d future rows of %s", corr, mask.sum(), zone)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Ensemble weight optimisation
# ══════════════════════════════════════════════════════════════════════════════

def _mape_objective(
    weights_arr: np.ndarray,
    xgb: np.ndarray,
    lgbm: np.ndarray,
    lstm: np.ndarray,
    actual: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    """
    Objective function ที่ scipy.optimize จะ minimize

    สูตร: MAPE = mean( |ensemble - actual| / (|actual| + epsilon) ) × 100

    ทำไมใช้ epsilon?
        ป้องกัน divide-by-zero ถ้า actual มีค่าน้อยมาก (< 1 MW)
        แต่ในทางปฏิบัติ scored rows ผ่านการกรอง actual > 1 MW มาแล้ว
    """
    w_xgb, w_lgbm, w_lstm = weights_arr

    # NaN-safe: แถวที่ LSTM=NaN → renormalize น้ำหนักอัตโนมัติ
    valid_lstm = ~np.isnan(lstm)
    denom_no_lstm = w_xgb + w_lgbm          # น้ำหนักรวมเมื่อ LSTM ไม่มีค่า

    ensemble = np.where(
        valid_lstm,
        xgb * w_xgb + lgbm * w_lgbm + lstm * w_lstm,           # มี LSTM
        xgb * (w_xgb / denom_no_lstm) + lgbm * (w_lgbm / denom_no_lstm),  # ไม่มี LSTM
    )

    ape = np.abs(ensemble - actual) / (np.abs(actual) + epsilon)
    return float(np.mean(ape) * 100)


def optimise_ensemble_weights(
    history_path: Path,
    fallback: dict[str, float],
) -> Optional[dict[str, float]]:
    """
    หา ensemble weights ที่ดีที่สุดจาก scored history ทั้งหมด

    Algorithm: SLSQP (Sequential Least Squares Programming)
    Constraints:
        - w_xgb + w_lgbm + w_lstm = 1   (weights บวกกันต้องได้ 1)
        - w_i ≥ 0 สำหรับทุก i           (น้ำหนักต้องไม่ติดลบ)
    Initial guess: weights ปัจจุบัน (fallback)

    ผลลัพธ์ถูกบันทึกลง optimal_weights.json และจะถูกโหลดใน run ถัดไป
    ไม่ได้ใช้ทันทีใน run นี้ — ให้ pipeline run เสร็จก่อน แล้วค่อยอัปเดต

    Returns None ถ้า:
        - ปิด flag ใน config (WEIGHT_OPT_ENABLED=False)
        - ยังมี scored rows ไม่พอ (< MIN_SCORED_FOR_WEIGHT_OPT)
        - scipy ไม่ได้ติดตั้ง
        - optimisation ไม่ converge
    """
    if not WEIGHT_OPT_ENABLED:
        return None

    try:
        from scipy.optimize import minimize
    except ImportError:
        logging.warning("scipy ไม่ได้ติดตั้ง — weight optimisation ข้ามไป (pip install scipy)")
        return None

    scored = _load_scored_history(history_path)
    if scored.empty:
        return None

    required_cols = ["pred_xgb", "pred_lgbm", "pred_lstm", "actual"]
    if not all(c in scored.columns for c in required_cols):
        return None

    n = len(scored)
    if n < MIN_SCORED_FOR_WEIGHT_OPT:
        logging.info(
            "Weight optimisation skipped — %d scored rows (ต้องการ ≥ %d)",
            n, MIN_SCORED_FOR_WEIGHT_OPT,
        )
        return None

    xgb_arr  = scored["pred_xgb"].to_numpy(dtype=float)
    lgbm_arr = scored["pred_lgbm"].to_numpy(dtype=float)
    lstm_arr = scored["pred_lstm"].to_numpy(dtype=float)
    act_arr  = scored["actual"].to_numpy(dtype=float)

    # ใช้ current weights เป็น starting point — ถ้า current weights ดีอยู่แล้ว
    # optimiser จะไม่เปลี่ยนมาก (converge เร็วกว่าเริ่มจาก [0.33, 0.33, 0.33])
    x0 = np.array([
        fallback.get("pred_xgb",  0.40),
        fallback.get("pred_lgbm", 0.40),
        fallback.get("pred_lstm", 0.20),
    ])

    result = minimize(
        _mape_objective,
        x0,
        args=(xgb_arr, lgbm_arr, lstm_arr, act_arr),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-6, "maxiter": 200},
    )

    if not result.success:
        logging.warning("Weight optimisation ไม่ converge: %s", result.message)
        return None

    w_xgb, w_lgbm, w_lstm = result.x
    optimal = {
        "pred_xgb":  round(float(w_xgb),  4),
        "pred_lgbm": round(float(w_lgbm), 4),
        "pred_lstm": round(float(w_lstm), 4),
    }

    # บันทึกลง disk พร้อม metadata เพื่อ traceability
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights":         optimal,
        "mape_at_optimum": round(float(result.fun), 4),
        "n_rows_used":     n,
        "updated_at":      datetime.now().isoformat(timespec="seconds"),
    }
    with open(OPTIMAL_WEIGHTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    logging.info(
        "Optimal weights updated | XGB=%.1f%% LGBM=%.1f%% LSTM=%.1f%% "
        "→ MAPE %.3f%% (%d scored rows)",
        w_xgb * 100, w_lgbm * 100, w_lstm * 100, result.fun, n,
    )
    _log_event(
        "WEIGHT_OPT_UPDATED", "ALL",
        f"xgb={w_xgb:.3f} lgbm={w_lgbm:.3f} lstm={w_lstm:.3f} "
        f"mape={result.fun:.3f} n={n}",
        float(result.fun),
    )
    return optimal


def load_optimal_weights(fallback: dict[str, float]) -> dict[str, float]:
    """
    โหลด weights ที่ optimize ไว้จาก disk

    ถ้าไม่มีไฟล์ หรือไฟล์ผิดรูปแบบ → ใช้ fallback (hardcoded weights)
    นี่คือวิธีที่ทำให้ run แรกสุดทำงานได้ โดยไม่ต้องมี history มาก่อน
    """
    if not OPTIMAL_WEIGHTS_PATH.exists():
        logging.info("optimal_weights.json ยังไม่มี → ใช้ default weights")
        return fallback

    try:
        with open(OPTIMAL_WEIGHTS_PATH) as f:
            data = json.load(f)
        weights = data.get("weights", {})

        # ตรวจว่า keys ตรงกัน — ป้องกัน schema เปลี่ยน
        if set(weights.keys()) == set(fallback.keys()):
            logging.info(
                "Loaded optimal weights: XGB=%.1f%% LGBM=%.1f%% LSTM=%.1f%% "
                "(MAPE %.3f%% จาก %d rows ณ %s)",
                weights["pred_xgb"]  * 100,
                weights["pred_lgbm"] * 100,
                weights["pred_lstm"] * 100,
                data.get("mape_at_optimum", float("nan")),
                data.get("n_rows_used", 0),
                data.get("updated_at", "unknown"),
            )
            return weights
    except Exception as exc:
        logging.warning("โหลด optimal_weights.json ไม่ได้: %s → ใช้ default weights", exc)

    return fallback


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAPE alerts
# ══════════════════════════════════════════════════════════════════════════════

def check_mape_alerts(metrics_df: pd.DataFrame) -> list[str]:
    """
    Log CRITICAL สำหรับภาคที่ Ensemble MAPE เกิน MAPE_ALERT_PCT
    Return: list ของ alert messages (ใช้ใน main.py เพื่อ summary log)
    """
    alerts: list[str] = []
    if metrics_df.empty or "MAPE" not in metrics_df.columns:
        return alerts

    zone_ensemble = metrics_df[
        (metrics_df["zone"] != "ALL")
        & (metrics_df["model"].str.contains("Ensemble_XGB_LGBM", na=False))
        & (metrics_df["MAPE"].notna())
    ]

    for _, row in zone_ensemble.iterrows():
        mape = float(row["MAPE"])
        if mape > MAPE_ALERT_PCT:
            bias = float(row.get("BIAS", float("nan")))
            msg  = (
                f"MAPE ALERT | {row['zone']} | "
                f"Ensemble MAPE={mape:.2f}% > {MAPE_ALERT_PCT}% | "
                f"BIAS={bias:.1f} MW"
            )
            logging.critical(msg)
            _log_event("MAPE_ALERT", str(row["zone"]),
                       f"MAPE={mape:.2f}% BIAS={bias:.1f}", mape)
            alerts.append(msg)

    return alerts
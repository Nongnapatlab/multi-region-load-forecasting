import logging
import numpy as np
import pandas as pd
import holidays

from config import DATE_COL, TARGET_COL

# [MLOps] กำหนดคอลัมน์ Lag ที่จะใช้สร้าง Rolling/Diff แทน Target จริงเพื่อป้องกัน Data Leakage
LAG_COL = "REQ_LAST168H"


def _fill_placeholder_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    แทนที่ค่า 0 ใน TARGET_COL (requirement) ที่เป็น placeholder ของวันอนาคต
    ด้วยค่าสุดท้ายที่รู้จริง (forward fill)

    ทำไมต้องทำก่อนสร้าง lag features?
    ──────────────────────────────────────────────────────────────────
    test set มีเพียง ~12 แถวที่มี actual จริง (เช้าวันที่รัน pipeline)
    ส่วนที่เหลืออีก 276 แถว requirement = 0 (placeholder วันอนาคต)

    ถ้าปล่อยไว้เป็น 0:
        LAG_1D ของวันพรุ่งนี้ = ค่า today = 0  ← ไม่ใช่ค่าจริง
        LAG_2D ของวันมะรืน   = LAG_1D ที่ได้ 0 มาอีกที
        → error สะสมทวีคูณ → เส้น prediction ดิ่งลงทุกวัน

    หลัง ffill:
        LAG_1D = ค่าจริงสุดท้ายที่รู้ (เช่น 9,700 MW ของ 06:00 วันนี้)
        → โมเดลเห็นค่าที่สมเหตุสมผล → ไม่เกิด drop

    NOTE: ffill ทำแค่ใน df copy — ไม่แก้ actual column ที่ใช้คำนวณ MAPE
    ──────────────────────────────────────────────────────────────────
    """
    df = df.copy()
    if TARGET_COL in df.columns:
        # แทนที่ 0 ด้วย NaN → ffill → bfill สำหรับแถวต้น
        df[TARGET_COL] = (
            df[TARGET_COL]
            .replace(0, float("nan"))
            .ffill()
            .bfill()
        )
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if LAG_COL in df.columns:
        df[f"{LAG_COL}_roll_mean_3"] = df[LAG_COL].rolling(3, min_periods=1).mean()
        df[f"{LAG_COL}_roll_mean_6"] = df[LAG_COL].rolling(6, min_periods=1).mean()
        df[f"{LAG_COL}_roll_std_6"]  = df[LAG_COL].rolling(6, min_periods=1).std().fillna(0)
    else:
        logging.warning("Lag column '%s' not found. Skipping rolling features.", LAG_COL)
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if LAG_COL in df.columns:
        df[f"{LAG_COL}_diff_1"]      = df[LAG_COL].diff().fillna(0)
        df[f"{LAG_COL}_pct_change_1"] = (
            df[LAG_COL].pct_change()
            .replace([float("inf"), float("-inf")], 0)
            .fillna(0)
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar & Holiday Features — ให้โมเดลแยกแยะวันทำงานกับวันหยุดได้
    (พฤติกรรมการใช้ไฟต่างกันชัดเจน)
    """
    df = df.copy()
    if DATE_COL not in df.columns:
        return df

    th_holidays = holidays.country_holidays("TH")
    dates = pd.to_datetime(df[DATE_COL])

    df["is_holiday"]      = dates.dt.date.map(lambda x: 1 if x in th_holidays else 0)
    df["is_weekend"]      = dates.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_pre_holiday"]  = df["is_holiday"].shift(-1).fillna(0).astype(int)
    df["is_post_holiday"] = df["is_holiday"].shift(1).fillna(0).astype(int)
    df["day_of_week"]     = dates.dt.dayofweek
    df["month"]           = dates.dt.month
    df["quarter"]         = dates.dt.quarter
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function สำหรับประกอบร่าง Features ทั้งหมดเข้าด้วยกัน"""
    # ── ① แก้ placeholder-zero ก่อน — ต้องอยู่บรรทัดแรกสุดเสมอ ───────────
    # add_rolling_features และ add_delta_features ใช้ LAG_COL ที่คำนวณจาก
    # requirement → ถ้า requirement = 0, rolling mean และ diff จะผิดทันที
    df = _fill_placeholder_zeros(df)
    # ── ② สร้าง features ตามปกติ ──────────────────────────────────────────
    df = add_rolling_features(df)
    df = add_delta_features(df)
    df = add_calendar_features(df)
    return df
import logging
import pandas as pd
import holidays

# ดึง DATE_COL จาก config เพื่อใช้ทำปฏิทิน
from config import DATE_COL

# [MLOps] กำหนดคอลัมน์ Lag ที่จะใช้สร้าง Rolling/Diff แทน Target จริงเพื่อป้องกัน Data Leakage
# เปลี่ยน "REQ_LAST168H" เป็นชื่อคอลัมน์ Lag ที่คุณมีจริงในไฟล์ข้อมูลของคุณ
LAG_COL = "REQ_LAST168H" 


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if LAG_COL in df.columns:
        df[f"{LAG_COL}_roll_mean_3"] = df[LAG_COL].rolling(3, min_periods=1).mean()
        df[f"{LAG_COL}_roll_mean_6"] = df[LAG_COL].rolling(6, min_periods=1).mean()
        df[f"{LAG_COL}_roll_std_6"] = df[LAG_COL].rolling(6, min_periods=1).std().fillna(0)
    else:
        logging.warning("Lag column '%s' not found. Skipping rolling features.", LAG_COL)
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if LAG_COL in df.columns:
        df[f"{LAG_COL}_diff_1"] = df[LAG_COL].diff().fillna(0)
        df[f"{LAG_COL}_pct_change_1"] = df[LAG_COL].pct_change().replace([float("inf"), float("-inf")], 0).fillna(0)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง Calendar & Holiday Features เพื่อให้โมเดลแยกแยะวันทำงานปกติ
    ออกจากวันหยุดยาวและวันหยุดสุดสัปดาห์ได้ (พฤติกรรมการใช้ไฟต่างกันชัดเจน)
    """
    df = df.copy()
    if DATE_COL in df.columns:
        # โหลดข้อมูลวันหยุดของประเทศไทย (Library นี้ครอบคลุมวันหยุดไทยได้แม่นยำ)
        th_holidays = holidays.country_holidays('TH')
        
        # มั่นใจว่าคอลัมน์วันที่เป็น datetime และดึงเฉพาะวันที่ (Date) ออกมาเทียบ
        dates = pd.to_datetime(df[DATE_COL])
        
        # 1. วันหยุดนักขัตฤกษ์ (1 = วันหยุด, 0 = วันทำงาน)
        df["is_holiday"] = dates.dt.date.map(lambda x: 1 if x in th_holidays else 0)
        
        # 2. วันหยุดสุดสัปดาห์ เสาร์-อาทิตย์ (1 = วันหยุด, 0 = วันทำงาน)
        df["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
        
        # 3. จับรอยต่อเทศกาล (วันก่อนหยุด และ วันหลังหยุด มักจะมีการใช้ไฟผิดปกติ)
        # ใช้ shift เพื่อดึงค่าของวันพรุ่งนี้/เมื่อวาน มาไว้ในแถวปัจจุบัน
        df["is_pre_holiday"] = df["is_holiday"].shift(-1).fillna(0).astype(int)
        df["is_post_holiday"] = df["is_holiday"].shift(1).fillna(0).astype(int)
        
        # 4. ฤดูกาล (Seasonality)
        df["day_of_week"] = dates.dt.dayofweek   # 0=จันทร์, 6=อาทิตย์
        df["month"] = dates.dt.month             # 1-12
        df["quarter"] = dates.dt.quarter         # 1-4
        
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function สำหรับประกอบร่าง Features ทั้งหมดเข้าด้วยกัน"""
    df = add_rolling_features(df)
    df = add_delta_features(df)
    df = add_calendar_features(df)
    return df
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
CACHE_DIR     = DATA_DIR / "cache"
OUTPUT_DIR    = BASE_DIR / "output"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR   = OUTPUT_DIR / "metrics"
LOGS_DIR      = OUTPUT_DIR / "logs"
WEIGHTS_DIR   = OUTPUT_DIR / "weights"   # เก็บ optimal_weights.json จาก auto-optimisation

DATE_COL      = "for_date"
TARGET_COL    = "requirement"
RANDOM_STATE  = 42
EPSILON       = 1e-6

# ── LSTM ──────────────────────────────────────────────────────────────────────
USE_LSTM = True
# ลดจาก 24 → 6:
#   warmup จะกินแค่ 5 แถวแรก (แทนที่จะเป็น 23) ทำให้ LSTM เริ่ม predict
#   ได้ตั้งแต่แถวที่ 6 — ซึ่งครอบคลุม scored rows ส่วนใหญ่ที่มี actual จริง
LSTM_SEQUENCE_LENGTH = 6
TEST_SIZE_FALLBACK   = 0.2

# ── ─────────────────────────────────────────────────────────────────────────
# Monitoring & Auto-adaptation thresholds
# ปรับตัวเลขด้านล่าง**โดยไม่แตะโค้ด**เลย — นี่คือ config ทั้งหมดที่ใช้ควบคุม
# พฤติกรรมของ auto_adapt.py
# ─────────────────────────────────────────────────────────────────────────────

# Distribution drift (z-score เทียบกับ train mean)
# กฎง่าย ๆ: |z| < 1.5 = ปกติ / 1.5-2.5 = เฝ้าระวัง / > 2.5 = แก้ด่วน
DRIFT_Z_WARN    = 1.5   # log WARNING
DRIFT_Z_RETRAIN = 2.5   # trigger bias correction อัตโนมัติ

# MAPE alert — ถ้าโมเดลใดภาคใด ensemble MAPE เกินค่านี้ → log critical
MAPE_ALERT_PCT  = 3.0

# Bias correction — ปรับ prediction อนาคตให้ชดเชย systematic error ที่เห็นแล้ว
# เหตุผลที่ต้องการ MIN_SCORED_ROWS: ถ้ามีข้อมูลน้อยเกินไป (เช่น แค่ 2 แถว)
# bias estimate ไม่น่าเชื่อถือ อาจแก้แล้วยิ่งแย่
BIAS_CORRECTION_ENABLED        = True
MIN_SCORED_FOR_BIAS_CORRECTION = 24   # ≈ 12 ชั่วโมงของข้อมูลจริง

# Ensemble weight optimisation ด้วย scipy.optimize
# ผลที่ได้ถูกบันทึกลง output/weights/optimal_weights.json
# แล้ว run ถัดไปจะโหลดมาใช้อัตโนมัติ — ไม่ต้องแก้โค้ด
WEIGHT_OPT_ENABLED            = True
MIN_SCORED_FOR_WEIGHT_OPT     = 48   # ≈ 1 วันเต็มของข้อมูลจริง

ZONES = {
    "CAC": {
        "train": "http://sothailand.com/centerimg/CAC_Lag123.csv",
        "test":  "http://sothailand.com/centerimg/CAC_test_Lag123.csv",
        "cache_train": "CAC_train_V3_cached.csv",
        "cache_test":  "CAC_test_V3_cached.csv",
        "prefix": "CAC",
    },
    "MAC": {
        "train": "http://sothailand.com/centerimg/MAC_Lag123.csv",
        "test":  "http://sothailand.com/centerimg/MAC_test_Lag123.csv",
        "cache_train": "MAC_train_V3_cached.csv",
        "cache_test":  "MAC_test_V3_cached.csv",
        "prefix": "MAC",
    },
    "NAC": {
        "train": "http://sothailand.com/centerimg/NAC_Lag123.csv",
        "test":  "http://sothailand.com/centerimg/NAC_test_Lag123.csv",
        "cache_train": "NAC_train_V3_cached.csv",
        "cache_test":  "NAC_test_V3_cached.csv",
        "prefix": "NAC",
    },
    "NEC": {
        "train": "http://sothailand.com/centerimg/NEC_Lag123.csv",
        "test":  "http://sothailand.com/centerimg/NEC_test_Lag123.csv",
        "cache_train": "NEC_train_V4_cached.csv",
        "cache_test":  "NEC_test_V4_cached.csv",
        "prefix": "NEC",
    },
    "SAC": {
        "train": "http://sothailand.com/centerimg/SAC_Lag123.csv",
        "test":  "http://sothailand.com/centerimg/SAC_test_Lag123.csv",
        "cache_train": "SAC_train_V3_cached.csv",
        "cache_test":  "SAC_test_V3_cached.csv",
        "prefix": "SAC",
    },
}

XGB_PARAMS = {
    "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.9,
    "objective": "reg:squarederror", "random_state": RANDOM_STATE,
}
LGBM_PARAMS = {
    "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.9,
    "objective": "regression", "random_state": RANDOM_STATE, "verbose": -1,
}
LSTM_PARAMS = {
    "sequence_length": LSTM_SEQUENCE_LENGTH,
    "epochs": 20, "batch_size": 32, "units": 64, "dropout": 0.2,
}

DIRECTORIES = [RAW_DIR, CACHE_DIR, PREDICTIONS_DIR, METRICS_DIR, LOGS_DIR, WEIGHTS_DIR]
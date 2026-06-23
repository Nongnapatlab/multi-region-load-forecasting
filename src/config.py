from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
LOGS_DIR = OUTPUT_DIR / "logs"

DATE_COL = "for_date"
TARGET_COL = "requirement"
RANDOM_STATE = 42
USE_LSTM = True
LSTM_SEQUENCE_LENGTH = 24
TEST_SIZE_FALLBACK = 0.2
EPSILON = 1e-6

ZONES = {
    "CAC": {
        "train": "http://sothailand.com/centerimg/CAC_Lag123.csv",
        "test": "http://sothailand.com/centerimg/CAC_test_Lag123.csv",
        "cache_train": "CAC_train_V3_cached.csv",
        "cache_test": "CAC_test_V3_cached.csv",
        "prefix": "CAC",
    },
    "MAC": {
        "train": "http://sothailand.com/centerimg/MAC_Lag123.csv",
        "test": "http://sothailand.com/centerimg/MAC_test_Lag123.csv",
        "cache_train": "MAC_train_V3_cached.csv",
        "cache_test": "MAC_test_V3_cached.csv",
        "prefix": "MAC",
    },
    "NAC": {
        "train": "http://sothailand.com/centerimg/NAC_Lag123.csv",
        "test": "http://sothailand.com/centerimg/NAC_test_Lag123.csv",
        "cache_train": "NAC_train_V3_cached.csv",
        "cache_test": "NAC_test_V3_cached.csv",
        "prefix": "NAC",
    },
    "NEC": {
        "train": "http://sothailand.com/centerimg/NEC_Lag123.csv",
        "test": "http://sothailand.com/centerimg/NEC_test_Lag123.csv",
        "cache_train": "NEC_train_V4_cached.csv",
        "cache_test": "NEC_test_V4_cached.csv",
        "prefix": "NEC",
    },
    "SAC": {
        "train": "http://sothailand.com/centerimg/SAC_Lag123.csv",
        "test": "http://sothailand.com/centerimg/SAC_test_Lag123.csv",
        "cache_train": "SAC_train_V3_cached.csv",
        "cache_test": "SAC_test_V3_cached.csv",
        "prefix": "SAC",
    },
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "regression",
    "random_state": RANDOM_STATE,
}

LSTM_PARAMS = {
    "sequence_length": LSTM_SEQUENCE_LENGTH,
    "epochs": 20,
    "batch_size": 32,
    "units": 64,
    "dropout": 0.2,
}

DIRECTORIES = [RAW_DIR, CACHE_DIR, PREDICTIONS_DIR, METRICS_DIR, LOGS_DIR]

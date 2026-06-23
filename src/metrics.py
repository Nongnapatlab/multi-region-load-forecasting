import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import EPSILON


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum(np.abs(y_true), EPSILON)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def mean_bias(y_true, y_pred):
    return float(np.mean(np.array(y_pred) - np.array(y_true)))


def evaluate_predictions(y_true, y_pred) -> dict:
    valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true = np.array(y_true)[valid_mask]
    y_pred = np.array(y_pred)[valid_mask]

    if len(y_true) == 0:
        return {"MAPE": np.nan, "MAE": np.nan, "RMSE": np.nan, "BIAS": np.nan}

    return {
        "MAPE": safe_mape(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "BIAS": mean_bias(y_true, y_pred),
    }


def add_absolute_percentage_error(df: pd.DataFrame, actual_col: str, pred_col: str, output_col: str) -> pd.DataFrame:
    df = df.copy()
    denom = np.maximum(np.abs(df[actual_col]), EPSILON)
    df[output_col] = (np.abs(df[actual_col] - df[pred_col]) / denom) * 100
    return df

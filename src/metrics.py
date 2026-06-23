import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import EPSILON

# Rows where actual demand is (near) zero are treated as "not yet realized"
# placeholders (e.g. future for_date rows in the test CSV that haven't
# happened yet), not real zero-load observations. Including them blows up
# MAPE because of the near-zero denominator, so they're excluded from MAPE/
# MAE/RMSE/BIAS scoring. This threshold should stay far below any real load
# value (zones here run from hundreds to ~13,000 MW).
ZERO_ACTUAL_THRESHOLD = 1.0


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum(np.abs(y_true), EPSILON)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def mean_bias(y_true, y_pred):
    return float(np.mean(np.array(y_pred) - np.array(y_true)))


def evaluate_predictions(y_true, y_pred) -> dict:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    valid_mask = ~(pd.isna(y_true_arr) | pd.isna(y_pred_arr))
    not_yet_realized_mask = np.abs(y_true_arr) < ZERO_ACTUAL_THRESHOLD
    score_mask = valid_mask & ~not_yet_realized_mask

    n_total = len(y_true_arr)
    n_excluded = int((valid_mask & not_yet_realized_mask).sum())
    n_scored = int(score_mask.sum())

    y_true_scored = y_true_arr[score_mask]
    y_pred_scored = y_pred_arr[score_mask]

    if n_scored == 0:
        return {
            "MAPE": np.nan, "MAE": np.nan, "RMSE": np.nan, "BIAS": np.nan,
            "n_rows_total": n_total, "n_rows_scored": 0, "n_rows_excluded_zero_actual": n_excluded,
        }

    return {
        "MAPE": safe_mape(y_true_scored, y_pred_scored),
        "MAE": mean_absolute_error(y_true_scored, y_pred_scored),
        "RMSE": np.sqrt(mean_squared_error(y_true_scored, y_pred_scored)),
        "BIAS": mean_bias(y_true_scored, y_pred_scored),
        "n_rows_total": n_total,
        "n_rows_scored": n_scored,
        "n_rows_excluded_zero_actual": n_excluded,
    }


def add_absolute_percentage_error(df: pd.DataFrame, actual_col: str, pred_col: str, output_col: str) -> pd.DataFrame:
    """Per-row APE. Rows with not-yet-realized (near-zero) actuals get NaN
    instead of a misleading huge percentage, so daily MAPE aggregations
    naturally skip them."""
    df = df.copy()
    denom = np.maximum(np.abs(df[actual_col]), EPSILON)
    ape = (np.abs(df[actual_col] - df[pred_col]) / denom) * 100
    df[output_col] = ape.where(df[actual_col].abs() >= ZERO_ACTUAL_THRESHOLD, np.nan)
    return df

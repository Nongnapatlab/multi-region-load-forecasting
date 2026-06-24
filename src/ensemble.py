import numpy as np
import pandas as pd


def simple_average_ensemble(df: pd.DataFrame, prediction_cols: list[str], output_col: str = "pred_ensemble") -> pd.DataFrame:
    df = df.copy()
    df[output_col] = df[prediction_cols].mean(axis=1)
    return df


def weighted_average_ensemble(df: pd.DataFrame, weights: dict[str, float], output_col: str = "pred_ensemble_weighted") -> pd.DataFrame:
    """Weighted average that renormalizes weights row-by-row over the columns
    that are not NaN for that row, so a stray NaN in one model's prediction
    (e.g. LSTM during sequence warm-up, or any model failing on an edge-case
    row) doesn't silently drag the ensemble toward zero."""
    df = df.copy()
    cols = list(weights.keys())

    weight_arr = np.array([weights[c] for c in cols], dtype=float)
    values = df[cols].to_numpy(dtype=float)
    valid_mask = ~np.isnan(values)

    row_weights = valid_mask * weight_arr[np.newaxis, :]
    row_weight_sums = row_weights.sum(axis=1)

    weighted_values = np.nan_to_num(values, nan=0.0) * row_weights
    weighted_sum = weighted_values.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(row_weight_sums > 0, weighted_sum / row_weight_sums, np.nan)

    df[output_col] = result
    return df

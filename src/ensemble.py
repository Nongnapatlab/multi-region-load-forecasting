import numpy as np
import pandas as pd


def simple_average_ensemble(df: pd.DataFrame, prediction_cols: list[str], output_col: str = "pred_ensemble") -> pd.DataFrame:
    df = df.copy()
    df[output_col] = df[prediction_cols].mean(axis=1)
    return df


def weighted_average_ensemble(df: pd.DataFrame, weights: dict[str, float], output_col: str = "pred_ensemble_weighted") -> pd.DataFrame:
    df = df.copy()
    total_weight = sum(weights.values())
    if total_weight == 0:
        df[output_col] = np.nan
        return df

    weighted_sum = 0
    for col, weight in weights.items():
        weighted_sum += df[col] * weight
    df[output_col] = weighted_sum / total_weight
    return df
